"""Risk Assessor: evaluate path length, compaction cost, and risk metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.candidates import CandidatePath
from src.config_loader import PlannerConfig, RiskFieldConfig
from src.fields import RiskField
from src.geometry import FieldGrid


@dataclass(frozen=True)
class PathAssessment:
    """Metrics computed by the Risk Assessor for one candidate."""

    candidate: CandidatePath
    path_length_m: float
    compaction_cost: float
    mean_risk: float
    max_risk: float
    coverage_rate: float
    num_turns: int
    objective_length: float
    objective_weighted: float
    objective_rb: float

    @property
    def angle_deg(self) -> float:
        return self.candidate.angle_deg


def path_length(waypoints: list[tuple[float, float]]) -> float:
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        total += abs(x1 - x0) + abs(y1 - y0)
    return total


def _densify_waypoints(
    waypoints: list[tuple[float, float]],
    spacing_m: float,
) -> list[tuple[float, float]]:
    """Insert points along segments for smoother risk integration."""
    if len(waypoints) < 2:
        return waypoints
    dense: list[tuple[float, float]] = [waypoints[0]]
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        seg_len = abs(x1 - x0) + abs(y1 - y0)
        n = max(1, int(np.ceil(seg_len / max(spacing_m, 0.5))))
        for i in range(1, n + 1):
            t = i / n
            dense.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return dense


def _count_turns(waypoints: list[tuple[float, float]], angle_threshold_deg: float) -> int:
    if len(waypoints) < 3:
        return 0
    threshold = np.deg2rad(angle_threshold_deg)
    turns = 0
    prev_dir: tuple[float, float] | None = None
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        direction = (np.sign(dx), np.sign(dy))
        if prev_dir is not None and direction != prev_dir:
            turns += 1
        prev_dir = direction
    return turns


def compute_coverage_rate(
    waypoints: list[tuple[float, float]],
    grid: FieldGrid,
    tool_radius_m: float,
) -> float:
    """Fraction of inner workable cells covered within tool radius."""
    if not waypoints:
        return 0.0

    visited = np.zeros((grid.ny, grid.nx), dtype=bool)
    yy, xx = np.meshgrid(grid.y_coords, grid.x_coords, indexing="ij")
    radius_sq = tool_radius_m**2

    for x, y in waypoints:
        dist_sq = (xx - x) ** 2 + (yy - y) ** 2
        visited |= dist_sq <= radius_sq

    inner = grid.inner_mask
    if not np.any(inner):
        return 0.0
    return float(visited[inner].mean())


def assess_candidate(
    candidate: CandidatePath,
    grid: FieldGrid,
    risk: RiskField,
    risk_cfg: RiskFieldConfig,
    planner_cfg: PlannerConfig,
    lambda_weighted: float,
    beta_rb: float,
) -> PathAssessment:
    """
    Risk Assessor: compute L, C, mean_risk, max_risk, coverage.

    Dynamic repeat traversal and turning penalties are applied during
    segment-wise risk integration (planning proxy, not physical model).
    """
    waypoints = _densify_waypoints(candidate.waypoints, planner_cfg.waypoint_spacing_m / 2)
    L = path_length(candidate.waypoints)

    visit_count = np.zeros((grid.ny, grid.nx), dtype=int)
    static_samples: list[float] = []
    static_weights: list[float] = []
    compaction = 0.0

    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        seg_len = abs(x1 - x0) + abs(y1 - y0)
        if seg_len < 1e-9:
            continue
        n = max(2, int(np.ceil(seg_len / 1.0)) + 1)
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        base_r = risk.sample_many(xs, ys, grid)

        for i in range(len(xs)):
            static_samples.append(float(base_r[i]))
            static_weights.append(seg_len / n)

        dynamic_r = base_r.copy()
        for i in range(len(xs)):
            ix, iy = grid.world_to_index(float(xs[i]), float(ys[i]))
            repeat_factor = 1.0 + risk_cfg.repeat_penalty * visit_count[iy, ix]
            dynamic_r[i] = min(1.0, dynamic_r[i] * repeat_factor)
            visit_count[iy, ix] += 1

        seg_dynamic = float(dynamic_r.mean())
        compaction += seg_len * seg_dynamic

    num_turns = _count_turns(candidate.waypoints, risk_cfg.turning_angle_deg)
    if num_turns > 0:
        turn_cost = num_turns * risk_cfg.turning_penalty * (compaction / max(L, 1.0))
        compaction += turn_cost

    if static_samples:
        weights_arr = np.array(static_weights, dtype=float)
        samples_arr = np.array(static_samples, dtype=float)
        mean_risk = float(np.average(samples_arr, weights=weights_arr))
        max_risk = float(samples_arr.max())
    else:
        mean_risk = 0.0
        max_risk = 0.0

    tool_radius = planner_cfg.swath_width_m / 2.0
    coverage = compute_coverage_rate(candidate.waypoints, grid, tool_radius)

    return PathAssessment(
        candidate=candidate,
        path_length_m=L,
        compaction_cost=compaction,
        mean_risk=mean_risk,
        max_risk=max_risk,
        coverage_rate=coverage,
        num_turns=num_turns,
        objective_length=L,
        objective_weighted=L + lambda_weighted * compaction,
        objective_rb=L + beta_rb * compaction,
    )


def assess_all_candidates(
    candidates: list[CandidatePath],
    grid: FieldGrid,
    risk: RiskField,
    risk_cfg: RiskFieldConfig,
    planner_cfg: PlannerConfig,
    lambda_weighted: float,
    beta_rb: float,
    min_coverage: float,
) -> list[PathAssessment]:
    """Assess candidates; adapt coverage threshold for small/irregular fields."""
    if not candidates:
        return []

    all_assessed = [
        assess_candidate(
            cand, grid, risk, risk_cfg, planner_cfg, lambda_weighted, beta_rb
        )
        for cand in candidates
    ]
    best_cov = max(a.coverage_rate for a in all_assessed)
    threshold = min_coverage
    if best_cov < min_coverage - 1e-6:
        threshold = max(0.75, best_cov - 0.02)

    return [a for a in all_assessed if a.coverage_rate >= threshold - 1e-6]
