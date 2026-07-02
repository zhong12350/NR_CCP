"""Candidate path generation: angle sweep, swaths, route-order optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString
from shapely.prepared import prep

from src.config_loader import PlannerConfig
from src.geometry import (
    FieldGrid,
    extract_linestrings,
    rotate_geometry,
    sample_line,
)

# Beyond this many swath segments, 2-opt refinement is skipped for tractability.
_MAX_SWATHS_FOR_2OPT = 150


@dataclass(frozen=True)
class CandidatePath:
    """One coverage candidate from a specific swath angle."""

    angle_deg: float
    waypoints: list[tuple[float, float]]
    swath_count: int


def _generate_swath_lines(
    grid: FieldGrid,
    angle_deg: float,
    swath_width_m: float,
) -> list[LineString]:
    """
    Generate swath centerline segments clipped to the inner workable polygon.

    For non-convex fields one probe line can intersect the polygon in several
    disjoint segments; all segments above a minimum length are kept so that
    concave regions are not silently dropped from coverage.
    """
    inner = grid.geometry.inner
    cx = inner.centroid.x
    cy = inner.centroid.y

    rotated_inner = rotate_geometry(inner, -angle_deg, (cx, cy))
    minx, miny, maxx, maxy = rotated_inner.bounds
    margin = swath_width_m
    min_seg_len = swath_width_m * 0.5

    swaths_world: list[LineString] = []
    x = minx + swath_width_m / 2.0
    while x <= maxx - swath_width_m / 2.0 + 1e-6:
        probe = LineString([(x, miny - margin), (x, maxy + margin)])
        clipped = rotated_inner.intersection(probe)
        for seg in extract_linestrings(clipped):
            if seg.length <= min_seg_len:
                continue
            unrotated = rotate_geometry(seg, angle_deg, (cx, cy))
            if isinstance(unrotated, LineString):
                swaths_world.append(unrotated)
        x += swath_width_m

    return swaths_world


def _polyline_length(pts: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        total += float(np.hypot(x1 - x0, y1 - y0))
    return total


def _segment_inside(
    a: tuple[float, float],
    b: tuple[float, float],
    outer_prep,
    step_m: float = 2.0,
) -> bool:
    """Check that the straight segment a->b stays inside the field boundary."""
    seg = LineString([a, b])
    if seg.length < 1e-9:
        return True
    n = max(2, int(np.ceil(seg.length / step_m)) + 1)
    for t in np.linspace(0.0, 1.0, n):
        p = seg.interpolate(t, normalized=True)
        if not outer_prep.covers(p):
            return False
    return True


def _bridge(
    a: tuple[float, float],
    b: tuple[float, float],
    grid: FieldGrid,
    outer_prep,
) -> list[tuple[float, float]]:
    """
    Connect two points with the shortest valid transition through the field.

    Tries the direct segment first, then L-shaped detours and boundary-edge
    routes; every candidate is validated along its full geometry (not only at
    vertices) against the outer field polygon.
    """
    if np.allclose(a, b, atol=1e-6):
        return []
    ax, ay = a
    bx, by = b

    candidates: list[list[tuple[float, float]]] = [
        [b],
        [(bx, ay), b],
        [(ax, by), b],
    ]

    minx, miny, maxx, maxy = grid.geometry.outer.bounds
    margin = grid.cell_size_m
    for y_edge in (miny + margin, maxy - margin):
        candidates.append([(ax, y_edge), (bx, y_edge), b])
    for x_edge in (minx + margin, maxx - margin):
        candidates.append([(x_edge, ay), (x_edge, by), b])

    def valid(path: list[tuple[float, float]]) -> bool:
        pts = [a] + path
        for p0, p1 in zip(pts[:-1], pts[1:]):
            if not _segment_inside(p0, p1, outer_prep):
                return False
        return True

    valid_cands = [c for c in candidates if valid(c)]
    if not valid_cands:
        return [b]
    return min(valid_cands, key=lambda c: _polyline_length([a] + c))


# --- Route-order optimization over swath segments -------------------------
#
# Decision variables per angle: the visiting order of swath segments and the
# travel direction of each segment. The serpentine order is the baseline;
# greedy direction assignment plus 2-opt refinement shortens transitions on
# concave/multi-segment fields where plain serpentine is suboptimal.


def _endpoints(swath: LineString) -> tuple[tuple[float, float], tuple[float, float]]:
    coords = list(swath.coords)
    return (
        (float(coords[0][0]), float(coords[0][1])),
        (float(coords[-1][0]), float(coords[-1][1])),
    )


def _serpentine_order(swaths: list[LineString], angle_deg: float) -> list[int]:
    """Order swath segments by perpendicular offset, then along-swath position."""
    rad = np.deg2rad(angle_deg)
    along = (np.cos(rad), np.sin(rad))
    perp = (-np.sin(rad), np.cos(rad))

    def key(idx: int) -> tuple[float, float]:
        mid = swaths[idx].interpolate(0.5, normalized=True)
        return (
            round(mid.x * perp[0] + mid.y * perp[1], 3),
            mid.x * along[0] + mid.y * along[1],
        )

    return sorted(range(len(swaths)), key=key)


def _transition_cost(
    exit_pt: tuple[float, float], entry_pt: tuple[float, float]
) -> float:
    return float(np.hypot(entry_pt[0] - exit_pt[0], entry_pt[1] - exit_pt[1]))


def _greedy_directions(
    order: list[int],
    endpoints: list[tuple[tuple[float, float], tuple[float, float]]],
    first_reversed: bool,
) -> tuple[list[bool], float]:
    """Assign travel directions greedily; return (reversed flags, transition cost)."""
    dirs: list[bool] = [first_reversed]
    total = 0.0
    p0, p1 = endpoints[order[0]]
    exit_pt = p0 if first_reversed else p1
    for idx in order[1:]:
        q0, q1 = endpoints[idx]
        d_fwd = _transition_cost(exit_pt, q0)
        d_rev = _transition_cost(exit_pt, q1)
        if d_rev < d_fwd:
            dirs.append(True)
            total += d_rev
            exit_pt = q0
        else:
            dirs.append(False)
            total += d_fwd
            exit_pt = q1
    return dirs, total


def _order_cost(
    order: list[int],
    endpoints: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[float, list[bool]]:
    """Best greedy-direction cost over both first-swath orientations."""
    dirs_a, cost_a = _greedy_directions(order, endpoints, first_reversed=False)
    dirs_b, cost_b = _greedy_directions(order, endpoints, first_reversed=True)
    if cost_b < cost_a:
        return cost_b, dirs_b
    return cost_a, dirs_a


def _two_opt(
    order: list[int],
    endpoints: list[tuple[tuple[float, float], tuple[float, float]]],
    max_passes: int,
) -> list[int]:
    """First-improvement 2-opt on the swath visiting order."""
    n = len(order)
    if n < 4:
        return order
    best_cost, _ = _order_cost(order, endpoints)
    for _ in range(max(max_passes, 0)):
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                trial = order[: i + 1] + order[i + 1 : j + 1][::-1] + order[j + 1 :]
                cost, _ = _order_cost(trial, endpoints)
                if cost < best_cost - 1e-9:
                    order = trial
                    best_cost = cost
                    improved = True
        if not improved:
            break
    return order


def _build_route(
    swaths: list[LineString],
    order: list[int],
    dirs: list[bool],
    spacing_m: float,
    grid: FieldGrid,
    outer_prep,
) -> list[tuple[float, float]]:
    """Assemble waypoints through ordered/oriented swaths with valid bridges."""
    waypoints: list[tuple[float, float]] = []
    current: tuple[float, float] | None = None

    for idx, reverse in zip(order, dirs):
        pts = sample_line(swaths[idx], spacing_m)
        if len(pts) < 2:
            continue
        if reverse:
            pts = list(reversed(pts))
        if current is not None:
            waypoints.extend(_bridge(current, pts[0], grid, outer_prep))
        waypoints.extend(pts)
        current = pts[-1]

    return waypoints


def generate_candidate(
    grid: FieldGrid,
    angle_deg: float,
    planner_cfg: PlannerConfig,
) -> CandidatePath | None:
    """Generate one candidate path for a given swath angle."""
    swaths = _generate_swath_lines(grid, angle_deg, planner_cfg.swath_width_m)
    if not swaths:
        return None

    endpoints = [_endpoints(s) for s in swaths]
    order = _serpentine_order(swaths, angle_deg)
    if (
        planner_cfg.route_optimizer == "2opt"
        and len(swaths) <= _MAX_SWATHS_FOR_2OPT
    ):
        order = _two_opt(order, endpoints, planner_cfg.route_2opt_passes)
    _, dirs = _order_cost(order, endpoints)

    outer_prep = prep(grid.geometry.outer.buffer(grid.cell_size_m * 0.5))
    waypoints = _build_route(
        swaths, order, dirs, planner_cfg.waypoint_spacing_m, grid, outer_prep
    )
    if len(waypoints) < 2:
        return None
    return CandidatePath(
        angle_deg=float(angle_deg), waypoints=waypoints, swath_count=len(swaths)
    )


def enumerate_candidates(
    grid: FieldGrid,
    planner_cfg: PlannerConfig,
) -> list[CandidatePath]:
    """Enumerate candidate paths over swath angles (5° step by default)."""
    candidates: list[CandidatePath] = []
    angle = 0.0
    while angle < 180.0:
        cand = generate_candidate(grid, angle, planner_cfg)
        if cand is not None:
            candidates.append(cand)
        angle += planner_cfg.angle_step_deg
    return candidates
