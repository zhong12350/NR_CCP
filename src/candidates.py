"""Candidate path generation: angle sweep, swaths, serpentine routes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point

from src.config_loader import PlannerConfig
from src.geometry import (
    FieldGrid,
    extract_linestrings,
    rotate_geometry,
    sample_line,
)


@dataclass(frozen=True)
class CandidatePath:
    """One coverage candidate from a specific swath angle."""

    angle_deg: float
    waypoints: list[tuple[float, float]]
    swath_count: int


def _longest_line(lines: list[LineString]) -> LineString | None:
    if not lines:
        return None
    return max(lines, key=lambda ln: ln.length)


def _generate_swath_lines(
    grid: FieldGrid,
    angle_deg: float,
    swath_width_m: float,
) -> list[LineString]:
    """Generate swath centerlines clipped to inner workable polygon."""
    inner = grid.geometry.inner
    cx = inner.centroid.x
    cy = inner.centroid.y

    rotated_inner = rotate_geometry(inner, -angle_deg, (cx, cy))
    minx, miny, maxx, maxy = rotated_inner.bounds
    margin = swath_width_m

    swaths_world: list[LineString] = []
    x = minx + swath_width_m / 2.0
    while x <= maxx - swath_width_m / 2.0 + 1e-6:
        probe = LineString([(x, miny - margin), (x, maxy + margin)])
        clipped = rotated_inner.intersection(probe)
        lines = extract_linestrings(clipped)
        longest = _longest_line(lines)
        if longest is not None and longest.length > swath_width_m * 0.5:
            unrotated = rotate_geometry(longest, angle_deg, (cx, cy))
            if isinstance(unrotated, LineString):
                swaths_world.append(unrotated)
        x += swath_width_m

    return swaths_world


def _sort_swaths(swaths: list[LineString], angle_deg: float, grid: FieldGrid) -> list[LineString]:
    """Order swaths along the direction perpendicular to swath lines."""
    inner = grid.geometry.inner
    cx, cy = inner.centroid.x, inner.centroid.y
    perp_rad = np.deg2rad(angle_deg + 90.0)

    def sort_key(line: LineString) -> float:
        mid = line.interpolate(0.5, normalized=True)
        dx = mid.x - cx
        dy = mid.y - cy
        return dx * np.cos(perp_rad) + dy * np.sin(perp_rad)

    return sorted(swaths, key=sort_key)


def _bridge_axis_aligned(
    a: tuple[float, float],
    b: tuple[float, float],
    grid: FieldGrid,
) -> list[tuple[float, float]]:
    """Connect two points with an axis-aligned path through headland when needed."""
    if np.allclose(a, b, atol=1e-6):
        return []
    ax, ay = a
    bx, by = b

    direct = [(bx, ay), b]
    candidates = [direct]

    minx, miny, maxx, maxy = grid.geometry.outer.bounds
    margin = grid.cell_size_m
    for y_edge in (miny + margin, maxy - margin):
        candidates.append([(ax, y_edge), (bx, y_edge), b])
    for x_edge in (minx + margin, maxx - margin):
        candidates.append([(x_edge, ay), (x_edge, by), b])

    def valid(path: list[tuple[float, float]]) -> bool:
        pts = [a] + path
        for p in pts:
            if not grid.geometry.outer.contains(Point(p)) and not grid.geometry.outer.touches(
                Point(p)
            ):
                return False
        return True

    valid_cands = [c for c in candidates if valid(c)]
    if not valid_cands:
        return direct
    return min(valid_cands, key=lambda c: _polyline_length([a] + c))


def _polyline_length(pts: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        total += abs(x1 - x0) + abs(y1 - y0)
    return total


def _serpentine_route(
    swaths: list[LineString],
    spacing_m: float,
    grid: FieldGrid,
) -> list[tuple[float, float]]:
    """Build serpentine waypoints through ordered swaths."""
    if not swaths:
        return []

    waypoints: list[tuple[float, float]] = []
    current: tuple[float, float] | None = None
    going_up = True

    for swath in swaths:
        pts = sample_line(swath, spacing_m)
        if len(pts) < 2:
            continue
        if not going_up:
            pts = list(reversed(pts))
        entry = pts[0]
        if current is not None:
            waypoints.extend(_bridge_axis_aligned(current, entry, grid))
        waypoints.extend(pts)
        current = pts[-1]
        going_up = not going_up

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
    ordered = _sort_swaths(swaths, angle_deg, grid)
    waypoints = _serpentine_route(ordered, planner_cfg.waypoint_spacing_m, grid)
    if len(waypoints) < 2:
        return None
    return CandidatePath(angle_deg=float(angle_deg), waypoints=waypoints, swath_count=len(ordered))


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
