"""Risk-aware candidate generation for NR-CCP."""

from __future__ import annotations

from src.assessor import PathAssessment
from src.candidates import CandidatePath, enumerate_candidates, generate_candidate
from src.config_loader import AppConfig, InformedSamplingConfig, PlannerConfig
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
    extra_angles: list[float] | None = None,
) -> list[CandidatePath]:
    """Generate candidates at informed angles plus optional extra angles."""
    if not sampling_cfg.enabled:
        return enumerate_candidates(grid, planner_cfg)

    sampler = InformedAngleSampler(sampling_cfg, seed=seed)
    angles = set(sampler.propose_angles(grid, risk, planner_cfg.swath_width_m))
    if extra_angles:
        for angle in extra_angles:
            angles.add(_angle_key(angle))
            for d in (-sampling_cfg.fine_step_deg, 0.0, sampling_cfg.fine_step_deg):
                angles.add(_angle_key(angle + d))

    candidates: list[CandidatePath] = []
    for angle in sorted(angles):
        cand = generate_candidate(grid, angle, planner_cfg)
        if cand is not None:
            candidates.append(cand)
    return candidates


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


def build_nr_ccp_pool(
    full_candidates: list[CandidatePath],
    full_assessments: list[PathAssessment],
    grid: FieldGrid,
    risk: RiskField,
    config: AppConfig,
    rb_ccp_assessment: PathAssessment | None = None,
) -> tuple[list[CandidatePath], list[PathAssessment]]:
    """
    Prior-guided adaptive pool for NR-CCP.

    Guarantees inclusion of:
    1) RB-CCP winner angle (if available)
    2) Top informed angles + local refinement
    3) Lowest proxy-risk coarse angles
    """
    extra_angles: list[float] = []
    if rb_ccp_assessment is not None:
        extra_angles.append(rb_ccp_assessment.angle_deg)

    if full_assessments:
        best_proxy = min(full_assessments, key=lambda a: (a.mean_risk, a.path_length_m))
        extra_angles.append(best_proxy.angle_deg)

    informed_cands = generate_informed_candidates(
        grid,
        risk,
        config.planner,
        config.informed_sampling,
        seed=config.seed,
        extra_angles=extra_angles,
    )

    # Always retain RB-CCP winner candidate object from full pool if present.
    merged = _merge_candidates(informed_cands)
    if rb_ccp_assessment is not None:
        rb_angle = _angle_key(rb_ccp_assessment.angle_deg)
        for cand in full_candidates:
            if _angle_key(cand.angle_deg) == rb_angle:
                merged = _merge_candidates(merged + [cand])
                break

    merged_angles = {_angle_key(c.angle_deg) for c in merged}
    informed_assess = [
        a for a in full_assessments if _angle_key(a.angle_deg) in merged_angles
    ]
    return merged, informed_assess


def build_candidate_pools(
    grid: FieldGrid,
    risk: RiskField,
    config: AppConfig,
    full_assessments: list[PathAssessment] | None = None,
    rb_ccp_assessment: PathAssessment | None = None,
) -> tuple[list[CandidatePath], list[CandidatePath], list[PathAssessment]]:
    """
    Return (full_pool, nr_ccp_pool, nr_ccp_assessments).

    Full pool is shared by baselines. NR-CCP uses a prior-guided subset that
    always covers the RB-CCP winner when assessments are available.
    """
    full = enumerate_candidates(grid, config.planner)
    if full_assessments is None:
        informed = generate_informed_candidates(
            grid, risk, config.planner, config.informed_sampling, seed=config.seed
        )
        return full, informed, []

    nr_cands, nr_assess = build_nr_ccp_pool(
        full, full_assessments, grid, risk, config, rb_ccp_assessment
    )
    return full, nr_cands, nr_assess
