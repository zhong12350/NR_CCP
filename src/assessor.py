"""Risk Assessor: evaluate path length, compaction cost, and risk metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.candidates import CandidatePath
from src.config_loader import PlannerConfig, RiskFieldConfig
from src.fields import RiskField
from src.geometry import FieldGrid
from src.physics import PhysicsFactors


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
    headland_cost: float = 0.0
    hotspot_cost: float = 0.0
    pass_count_cost: float = 0.0
    repeat_cost: float = 0.0
    turn_cost: float = 0.0
    wheel_load_n: float = 0.0
    contact_pressure_kpa: float = 0.0
    load_factor: float = 1.0
    pressure_factor: float = 1.0
    moisture_factor: float = 1.0
    physics_factor: float = 1.0

    @property
    def angle_deg(self) -> float:
        return self.candidate.angle_deg


@dataclass(frozen=True)
class FeasibilityCertificate:
    """Fallback / feasibility diagnostics for one field."""

    field_name: str
    delta: float
    num_full_candidates: int
    num_nr_candidates: int
    best_coverage: float
    min_mean_risk: float
    min_violation: float
    field_area_m2: float
    aspect_ratio: float
    boundary_complexity: float


def path_length(waypoints: list[tuple[float, float]]) -> float:
    """Euclidean route length."""
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        total += float(np.hypot(x1 - x0, y1 - y0))
    return total


def _densify_waypoints(
    waypoints: list[tuple[float, float]],
    spacing_m: float,
) -> list[tuple[float, float]]:
    if len(waypoints) < 2:
        return waypoints
    dense: list[tuple[float, float]] = [waypoints[0]]
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        seg_len = float(np.hypot(x1 - x0, y1 - y0))
        n = max(1, int(np.ceil(seg_len / max(spacing_m, 0.5))))
        for i in range(1, n + 1):
            t = i / n
            dense.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return dense


def _count_turns(waypoints: list[tuple[float, float]], angle_threshold_deg: float) -> int:
    """Count heading changes exceeding the threshold angle."""
    if len(waypoints) < 3:
        return 0
    threshold_rad = np.deg2rad(max(angle_threshold_deg, 1.0))
    turns = 0
    prev_heading: float | None = None
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        dx, dy = x1 - x0, y1 - y0
        if np.hypot(dx, dy) < 1e-9:
            continue
        heading = float(np.arctan2(dy, dx))
        if prev_heading is not None:
            diff = abs(heading - prev_heading)
            diff = min(diff, 2.0 * np.pi - diff)
            if diff > threshold_rad:
                turns += 1
        prev_heading = heading
    return turns


def compute_coverage_rate(
    waypoints: list[tuple[float, float]],
    grid: FieldGrid,
    tool_radius_m: float,
) -> float:
    """Exact polygon coverage: swept tool area intersected with inner workable area."""
    if len(waypoints) < 2:
        return 0.0
    inner = grid.geometry.inner
    if inner.is_empty or inner.area <= 0.0:
        return 0.0

    from shapely.geometry import LineString

    route = LineString(waypoints)
    if route.length < 1e-9:
        return 0.0
    swept = route.buffer(tool_radius_m)
    covered = swept.intersection(inner).area
    return float(covered / inner.area)


def assess_candidate(
    candidate: CandidatePath,
    grid: FieldGrid,
    risk: RiskField,
    risk_cfg: RiskFieldConfig,
    planner_cfg: PlannerConfig,
    lambda_weighted: float,
    beta_rb: float,
    physics_factors: PhysicsFactors | None = None,
) -> PathAssessment:
    """Risk Assessor with decomposed compaction and physics-informed factors."""
    waypoints = _densify_waypoints(candidate.waypoints, planner_cfg.waypoint_spacing_m / 2)
    L = path_length(candidate.waypoints)
    physics = physics_factors.combined_factor if physics_factors else 1.0

    visit_count = np.zeros((grid.ny, grid.nx), dtype=int)
    static_samples: list[float] = []
    static_weights: list[float] = []
    compaction = 0.0
    headland_cost = 0.0
    hotspot_cost = 0.0
    pass_count_cost = 0.0
    repeat_cost = 0.0

    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        seg_len = float(np.hypot(x1 - x0, y1 - y0))
        if seg_len < 1e-9:
            continue
        n = max(2, int(np.ceil(seg_len / 1.0)) + 1)
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        base_r = risk.sample_many(xs, ys, grid) * physics
        head_r = risk.sample_layer_many(risk.headland_layer, xs, ys, grid) * physics
        hot_r = risk.sample_layer_many(risk.hotspot_layer, xs, ys, grid) * physics
        pass_r = risk.sample_layer_many(risk.pass_count_layer, xs, ys, grid) * physics

        for i in range(len(xs)):
            static_samples.append(float(min(1.0, base_r[i])))
            static_weights.append(seg_len / n)

        dynamic_r = base_r.copy()
        for i in range(len(xs)):
            ix, iy = grid.world_to_index(float(xs[i]), float(ys[i]))
            repeat_factor = 1.0 + risk_cfg.repeat_penalty * visit_count[iy, ix]
            added_repeat = max(0.0, dynamic_r[i] * (repeat_factor - 1.0))
            dynamic_r[i] = min(1.0, dynamic_r[i] * repeat_factor)
            repeat_cost += (seg_len / n) * added_repeat
            visit_count[iy, ix] += 1

        compaction += seg_len * float(dynamic_r.mean())
        headland_cost += seg_len * float(head_r.mean())
        hotspot_cost += seg_len * float(hot_r.mean())
        pass_count_cost += seg_len * float(pass_r.mean())

    num_turns = _count_turns(candidate.waypoints, risk_cfg.turning_angle_deg)
    turn_cost = 0.0
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
        headland_cost=headland_cost,
        hotspot_cost=hotspot_cost,
        pass_count_cost=pass_count_cost,
        repeat_cost=repeat_cost,
        turn_cost=turn_cost,
        wheel_load_n=physics_factors.wheel_load_n if physics_factors else 0.0,
        contact_pressure_kpa=physics_factors.contact_pressure_kpa if physics_factors else 0.0,
        load_factor=physics_factors.load_factor if physics_factors else 1.0,
        pressure_factor=physics_factors.pressure_factor if physics_factors else 1.0,
        moisture_factor=physics_factors.moisture_factor if physics_factors else 1.0,
        physics_factor=physics,
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
    physics_factors: PhysicsFactors | None = None,
) -> list[PathAssessment]:
    if not candidates:
        return []

    all_assessed = [
        assess_candidate(
            cand,
            grid,
            risk,
            risk_cfg,
            planner_cfg,
            lambda_weighted,
            beta_rb,
            physics_factors=physics_factors,
        )
        for cand in candidates
    ]
    return filter_by_coverage(all_assessed, min_coverage)


def filter_by_coverage(
    assessments: list[PathAssessment],
    min_coverage: float,
) -> list[PathAssessment]:
    """
    Strict coverage constraint: keep candidates with coverage >= min_coverage.

    If no candidate satisfies the constraint, the field is coverage-infeasible
    at this configuration; return only the best-coverage candidates so that
    downstream selection still produces a plan, and let the recorded
    coverage_rate (< min_coverage) expose the violation. The threshold is
    never silently relaxed.
    """
    if not assessments:
        return []
    feasible = [a for a in assessments if a.coverage_rate >= min_coverage - 1e-9]
    if feasible:
        return feasible
    best_cov = max(a.coverage_rate for a in assessments)
    return [a for a in assessments if a.coverage_rate >= best_cov - 1e-6]


def build_feasibility_certificate(
    field_name: str,
    grid: FieldGrid,
    full_assessments: list[PathAssessment],
    nr_assessments: list[PathAssessment],
    delta: float,
) -> FeasibilityCertificate:
    pool = full_assessments or nr_assessments
    if not pool:
        return FeasibilityCertificate(
            field_name=field_name,
            delta=delta,
            num_full_candidates=0,
            num_nr_candidates=len(nr_assessments),
            best_coverage=0.0,
            min_mean_risk=1.0,
            min_violation=1.0,
            field_area_m2=grid.geometry.area_m2,
            aspect_ratio=grid.geometry.aspect_ratio,
            boundary_complexity=float(grid.geometry.outer.length / max(grid.geometry.area_m2, 1.0)),
        )

    min_risk = min(a.mean_risk for a in pool)
    return FeasibilityCertificate(
        field_name=field_name,
        delta=delta,
        num_full_candidates=len(full_assessments),
        num_nr_candidates=len(nr_assessments),
        best_coverage=max(a.coverage_rate for a in pool),
        min_mean_risk=min_risk,
        min_violation=max(0.0, min_risk - delta),
        field_area_m2=grid.geometry.area_m2,
        aspect_ratio=grid.geometry.aspect_ratio,
        boundary_complexity=float(grid.geometry.outer.length / max(grid.geometry.area_m2, 1.0)),
    )
