"""Risk-aware candidate generation for NR-CCP.

The NR-CCP informed pool is generated and assessed independently of the full
enumeration pool. It never peeks at full-pool assessments or at the RB-CCP
winner, so the RB-CCP vs NR-CCP comparison measures what the informed prior
actually buys: fewer generated/assessed candidates for comparable solution
quality.
"""

from __future__ import annotations

from src.candidates import CandidatePath, enumerate_candidates, generate_candidate
from src.config_loader import InformedSamplingConfig, PlannerConfig
from src.fields import RiskField
from src.geometry import FieldGrid
from src.informed_sampling import InformedAngleSampler


def _angle_key(angle_deg: float) -> float:
    return round(angle_deg % 180.0, 2)


def _merge_candidates(candidates: list[CandidatePath]) -> list[CandidatePath]:
    """Deduplicate candidates by swath angle."""
    by_angle: dict[float, CandidatePath] = {}
    for cand in candidates:
        key = _angle_key(cand.angle_deg)
        if key not in by_angle:
            by_angle[key] = cand
    return sorted(by_angle.values(), key=lambda c: c.angle_deg)


def generate_informed_candidates(
    grid: FieldGrid,
    risk: RiskField,
    planner_cfg: PlannerConfig,
    sampling_cfg: InformedSamplingConfig,
    seed: int = 42,
) -> list[CandidatePath]:
    """Generate candidates at angles proposed by the informed prior only."""
    if not sampling_cfg.enabled:
        return enumerate_candidates(grid, planner_cfg)

    sampler = InformedAngleSampler(sampling_cfg, seed=seed)
    angles = sorted({_angle_key(a) for a in sampler.propose_angles(grid, risk, planner_cfg.swath_width_m)})

    candidates: list[CandidatePath] = []
    for angle in angles:
        cand = generate_candidate(grid, angle, planner_cfg)
        if cand is not None:
            candidates.append(cand)
    return _merge_candidates(candidates)


def rank_candidates_by_prior(
    candidates: list[CandidatePath],
    grid: FieldGrid,
    risk: RiskField,
    planner_cfg: PlannerConfig,
    sampling_cfg: InformedSamplingConfig,
    seed: int = 42,
) -> list[CandidatePath]:
    """Rank an existing candidate pool by the NR learned/heuristic prior."""
    if not candidates:
        return []

    sampler = InformedAngleSampler(sampling_cfg, seed=seed)
    scored = [
        (
            sampler.score_angle(grid, risk, cand.angle_deg, planner_cfg.swath_width_m),
            -cand.swath_count,
            cand,
        )
        for cand in candidates
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [cand for _, _, cand in scored]
