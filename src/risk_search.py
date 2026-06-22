"""Risk-aware candidate generation for NR-CCP."""

from __future__ import annotations

from src.candidates import CandidatePath, enumerate_candidates, generate_candidate
from src.config_loader import AppConfig, InformedSamplingConfig, PlannerConfig
from src.fields import RiskField
from src.geometry import FieldGrid
from src.informed_sampling import InformedAngleSampler


def generate_informed_candidates(
    grid: FieldGrid,
    risk: RiskField,
    planner_cfg: PlannerConfig,
    sampling_cfg: InformedSamplingConfig,
    seed: int = 42,
) -> list[CandidatePath]:
    """Generate candidates via informed angle sampling + local refinement."""
    if not sampling_cfg.enabled:
        return enumerate_candidates(grid, planner_cfg)

    sampler = InformedAngleSampler(sampling_cfg, seed=seed)
    angles = sampler.propose_angles(grid, risk, planner_cfg.swath_width_m)
    candidates: list[CandidatePath] = []
    for angle in angles:
        cand = generate_candidate(grid, angle, planner_cfg)
        if cand is not None:
            candidates.append(cand)
    return candidates


def build_candidate_pools(
    grid: FieldGrid,
    risk: RiskField,
    config: AppConfig,
) -> tuple[list[CandidatePath], list[CandidatePath]]:
    """
    Return (shared_full_pool, informed_pool).

    Baselines use the full enumerated pool; NR-CCP uses informed search pool.
    """
    full = enumerate_candidates(grid, config.planner)
    informed = generate_informed_candidates(
        grid,
        risk,
        config.planner,
        config.informed_sampling,
        seed=config.seed,
    )
    return full, informed
