"""Evaluation metrics for coverage paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.fields import RiskField
from src.geometry import FieldGrid
from src.planner import PlanResult


@dataclass
class PathMetrics:
    method: str
    path_length_m: float
    num_waypoints: int
    mean_risk: float
    max_risk: float
    risk_length_cost: float
    coverage_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def path_length(waypoints: list[tuple[float, float]]) -> float:
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        total += abs(x1 - x0) + abs(y1 - y0)
    return total


def compute_coverage_rate(
    waypoints: list[tuple[float, float]],
    grid: FieldGrid,
    tool_radius_m: float,
) -> float:
    """Fraction of free cells visited within tool_radius of any waypoint."""
    if not waypoints:
        return 0.0

    visited = np.zeros((grid.ny, grid.nx), dtype=bool)
    yy, xx = np.meshgrid(grid.y_coords, grid.x_coords, indexing="ij")
    radius_sq = tool_radius_m**2

    for x, y in waypoints:
        dist_sq = (xx - x) ** 2 + (yy - y) ** 2
        visited |= dist_sq <= radius_sq

    free = ~grid.obstacle_mask
    if not np.any(free):
        return 0.0
    return float(visited[free].mean())


def evaluate_plan(
    result: PlanResult,
    grid: FieldGrid,
    risk: RiskField,
    tool_radius_m: float,
) -> PathMetrics:
    risks = risk.sample_path(result.waypoints, grid)
    seg_lengths = []
    for (x0, y0), (x1, y1) in zip(result.waypoints[:-1], result.waypoints[1:]):
        seg_lengths.append(abs(x1 - x0) + abs(y1 - y0))
    seg_lengths_arr = np.array(seg_lengths, dtype=float) if seg_lengths else np.array([0.0])
    seg_risks = (risks[:-1] + risks[1:]) / 2 if len(risks) > 1 else risks

    if len(seg_risks) != len(seg_lengths_arr):
        risk_length_cost = float(np.sum(risks) * (path_length(result.waypoints) / max(len(risks), 1)))
    else:
        risk_length_cost = float(np.sum(seg_lengths_arr * seg_risks))

    return PathMetrics(
        method=result.method,
        path_length_m=path_length(result.waypoints),
        num_waypoints=len(result.waypoints),
        mean_risk=float(risks.mean()) if len(risks) else 0.0,
        max_risk=float(risks.max()) if len(risks) else 0.0,
        risk_length_cost=risk_length_cost,
        coverage_rate=compute_coverage_rate(result.waypoints, grid, tool_radius_m),
    )


def evaluate_all(
    results: list[PlanResult],
    grid: FieldGrid,
    risk: RiskField,
    tool_radius_m: float,
) -> list[PathMetrics]:
    return [evaluate_plan(r, grid, risk, tool_radius_m) for r in results]
