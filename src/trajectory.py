"""Post-process coverage waypoints into bounded-curvature executable trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleMotionConfig:
    """Kinematic limits for agricultural vehicle motion."""

    min_turn_radius_m: float = 5.0
    swath_width_m: float = 6.0
    wheelbase_m: float = 2.5
    body_length_m: float = 3.5
    body_width_m: float = 2.0
    cruise_speed_mps: float = 2.0
    sample_spacing_m: float = 0.35


@dataclass(frozen=True)
class ExecutableTrajectory:
    xs: np.ndarray
    ys: np.ndarray
    yaws: np.ndarray
    curvatures: np.ndarray
    arc_lengths: np.ndarray
    is_turn: np.ndarray
    geo_waypoints: list[tuple[float, float]]


@dataclass(frozen=True)
class ExecutableMetrics:
    path_length_geo_m: float
    path_length_exec_m: float
    num_turns: int
    max_curvature: float
    turning_length_m: float
    length_increase_pct: float


def _dedupe_points(
    waypoints: list[tuple[float, float]], tol: float = 1e-4
) -> list[tuple[float, float]]:
    if not waypoints:
        return []
    out = [waypoints[0]]
    for pt in waypoints[1:]:
        if np.hypot(pt[0] - out[-1][0], pt[1] - out[-1][1]) > tol:
            out.append(pt)
    return out


def _segment_length(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))


def _polyline_length(pts: list[tuple[float, float]]) -> float:
    return sum(_segment_length(a, b) for a, b in zip(pts[:-1], pts[1:]))


def _normalize(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _sample_arc(
    center: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    spacing_m: float,
    turn_left: bool,
) -> list[tuple[float, float]]:
    v0 = start - center
    v1 = end - center
    a0 = float(np.arctan2(v0[1], v0[0]))
    a1 = float(np.arctan2(v1[1], v1[0]))
    if turn_left:
        while a1 <= a0:
            a1 += 2.0 * np.pi
        sweep = a1 - a0
    else:
        while a1 >= a0:
            a1 -= 2.0 * np.pi
        sweep = a1 - a0

    arc_len = abs(sweep) * radius
    n = max(2, int(np.ceil(arc_len / max(spacing_m, 0.1))) + 1)
    angles = np.linspace(a0, a1, n)
    return [
        (float(center[0] + radius * np.cos(a)), float(center[1] + radius * np.sin(a)))
        for a in angles
    ]


def _fillet_polyline(
    waypoints: list[tuple[float, float]],
    radius: float,
    spacing_m: float,
    angle_threshold_deg: float = 12.0,
) -> list[tuple[float, float]]:
    """Replace sharp corners with circular-arc fillets (bounded curvature)."""
    pts = _dedupe_points(waypoints)
    if len(pts) < 3:
        return pts

    out: list[tuple[float, float]] = [pts[0]]
    threshold = np.deg2rad(angle_threshold_deg)

    for i in range(1, len(pts) - 1):
        p0 = np.array(pts[i - 1], dtype=float)
        p1 = np.array(pts[i], dtype=float)
        p2 = np.array(pts[i + 1], dtype=float)

        v_in = _normalize(p0 - p1)
        v_out = _normalize(p2 - p1)
        if v_in is None or v_out is None:
            out.append(pts[i])
            continue

        cos_angle = float(np.clip(np.dot(v_in, v_out), -1.0, 1.0))
        turn_angle = float(np.arccos(cos_angle))
        if turn_angle < threshold:
            out.append(pts[i])
            continue

        leg_in = float(np.linalg.norm(p1 - p0))
        leg_out = float(np.linalg.norm(p2 - p1))
        tan_dist = radius / max(np.tan(turn_angle / 2.0), 1e-6)
        tan_dist = min(tan_dist, 0.45 * leg_in, 0.45 * leg_out)
        if tan_dist < 0.05:
            out.append(pts[i])
            continue

        t_start = p1 + v_in * tan_dist
        t_end = p1 + v_out * tan_dist
        bisector = _normalize(v_in + v_out)
        if bisector is None:
            out.append(pts[i])
            continue
        center_dist = radius / max(np.sin(turn_angle / 2.0), 1e-6)
        center = p1 + bisector * center_dist

        cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
        turn_left = cross > 0.0
        arc_pts = _sample_arc(center, t_start, t_end, radius, spacing_m, turn_left)
        if out and arc_pts:
            if np.hypot(out[-1][0] - arc_pts[0][0], out[-1][1] - arc_pts[0][1]) < 1e-4:
                arc_pts = arc_pts[1:]
        out.extend(arc_pts)

    out.append(pts[-1])
    return out


def _resample_uniform(
    pts: list[tuple[float, float]], spacing_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(pts) < 2:
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        return xs, ys, np.zeros(len(xs))

    seg_lens = [_segment_length(a, b) for a, b in zip(pts[:-1], pts[1:])]
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-6:
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        return xs, ys, np.zeros(len(xs))

    samples = np.arange(0.0, total + spacing_m * 0.5, spacing_m)
    samples = samples[samples <= total + 1e-9]
    if samples[-1] < total - 1e-6:
        samples = np.append(samples, total)

    xs_out: list[float] = []
    ys_out: list[float] = []
    seg_idx = 0
    for s in samples:
        while seg_idx < len(seg_lens) - 1 and cum[seg_idx + 1] < s - 1e-9:
            seg_idx += 1
        seg_start = cum[seg_idx]
        seg_len = max(seg_lens[seg_idx], 1e-9)
        t = (s - seg_start) / seg_len
        x0, y0 = pts[seg_idx]
        x1, y1 = pts[seg_idx + 1]
        xs_out.append(x0 + t * (x1 - x0))
        ys_out.append(y0 + t * (y1 - y0))

    return np.array(xs_out), np.array(ys_out), samples


def _compute_yaw_curvature(
    xs: np.ndarray, ys: np.ndarray, arc_lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    yaws = np.zeros(n)
    curvatures = np.zeros(n)
    if n < 2:
        return yaws, curvatures

    for i in range(n):
        i0 = max(0, i - 1)
        i1 = min(n - 1, i + 1)
        dx = xs[i1] - xs[i0]
        dy = ys[i1] - ys[i0]
        yaws[i] = float(np.arctan2(dy, dx))

    for i in range(1, n - 1):
        dyaw = yaws[i + 1] - yaws[i - 1]
        dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi
        ds = max(arc_lengths[i + 1] - arc_lengths[i - 1], 1e-6)
        curvatures[i] = dyaw / ds

    curvatures[0] = curvatures[1] if n > 2 else 0.0
    curvatures[-1] = curvatures[-2] if n > 2 else 0.0
    return yaws, curvatures


def postprocess_executable(
    waypoints: list[tuple[float, float]],
    motion: VehicleMotionConfig,
) -> ExecutableTrajectory:
    """Convert geometric coverage waypoints into a smooth executable trajectory."""
    geo = _dedupe_points(waypoints)
    filleted = _fillet_polyline(
        geo,
        radius=motion.min_turn_radius_m,
        spacing_m=motion.sample_spacing_m,
    )
    xs, ys, arc_lengths = _resample_uniform(filleted, motion.sample_spacing_m)
    yaws, curvatures = _compute_yaw_curvature(xs, ys, arc_lengths)
    kappa_thr = 0.5 / max(motion.min_turn_radius_m, 1e-3)
    is_turn = np.abs(curvatures) > kappa_thr
    return ExecutableTrajectory(
        xs=xs,
        ys=ys,
        yaws=yaws,
        curvatures=curvatures,
        arc_lengths=arc_lengths,
        is_turn=is_turn,
        geo_waypoints=geo,
    )


def compute_executable_metrics(
    traj: ExecutableTrajectory,
    motion: VehicleMotionConfig,
) -> ExecutableMetrics:
    geo_len = _polyline_length(traj.geo_waypoints)
    exec_len = float(traj.arc_lengths[-1]) if len(traj.arc_lengths) else 0.0
    kappa_thr = 0.5 / max(motion.min_turn_radius_m, 1e-3)
    turn_mask = np.abs(traj.curvatures) > kappa_thr
    turning_length = 0.0
    if len(traj.arc_lengths) > 1:
        ds = np.diff(traj.arc_lengths)
        turning_length = float(ds[turn_mask[1:]].sum())

    yaw_diff = np.diff(traj.yaws)
    yaw_diff = (yaw_diff + np.pi) % (2.0 * np.pi) - np.pi
    num_turns = int(np.sum(np.abs(yaw_diff) > np.deg2rad(45.0)))

    increase = 100.0 * (exec_len - geo_len) / max(geo_len, 1e-6)
    return ExecutableMetrics(
        path_length_geo_m=geo_len,
        path_length_exec_m=exec_len,
        num_turns=num_turns,
        max_curvature=float(np.max(np.abs(traj.curvatures))) if len(traj.curvatures) else 0.0,
        turning_length_m=turning_length,
        length_increase_pct=increase,
    )
