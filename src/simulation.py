"""Simulate vehicle motion along an executable trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.fields import RiskField
from src.geometry import FieldGrid
from src.trajectory import ExecutableTrajectory, VehicleMotionConfig


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw: float
    kappa: float
    s: float
    is_turn: bool
    risk: float
    progress: float


@dataclass(frozen=True)
class SimulationResult:
    states: list[VehicleState]
    mean_risk: float
    max_risk: float
    mean_risk_turn: float
    duration_s: float


def _interp_at_s(
    traj: ExecutableTrajectory, s: float
) -> tuple[float, float, float, float, bool]:
    s = float(np.clip(s, 0.0, traj.arc_lengths[-1] if len(traj.arc_lengths) else 0.0))
    idx = int(np.searchsorted(traj.arc_lengths, s, side="right") - 1)
    idx = int(np.clip(idx, 0, len(traj.xs) - 2))
    s0 = float(traj.arc_lengths[idx])
    s1 = float(traj.arc_lengths[idx + 1])
    t = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)

    x = float(traj.xs[idx] + t * (traj.xs[idx + 1] - traj.xs[idx]))
    y = float(traj.ys[idx] + t * (traj.ys[idx + 1] - traj.ys[idx]))
    yaw = float(traj.yaws[idx] + t * (traj.yaws[idx + 1] - traj.yaws[idx]))
    kappa = float(traj.curvatures[idx] + t * (traj.curvatures[idx + 1] - traj.curvatures[idx]))
    is_turn = bool(traj.is_turn[idx] or traj.is_turn[idx + 1])
    return x, y, yaw, kappa, is_turn


def simulate_along_trajectory(
    traj: ExecutableTrajectory,
    risk: RiskField,
    grid: FieldGrid,
    motion: VehicleMotionConfig,
    dt: float = 0.05,
) -> SimulationResult:
    """Advance the vehicle at constant speed along arc length s."""
    total_s = float(traj.arc_lengths[-1]) if len(traj.arc_lengths) else 0.0
    if total_s < 1e-6:
        return SimulationResult([], 0.0, 0.0, 0.0, 0.0)

    v = motion.cruise_speed_mps
    times = np.arange(0.0, total_s / max(v, 1e-6) + dt * 0.5, dt)
    states: list[VehicleState] = []
    risks: list[float] = []
    turn_risks: list[float] = []

    for t in times:
        s = min(v * t, total_s)
        x, y, yaw, kappa, is_turn = _interp_at_s(traj, s)
        r = float(risk.sample(x, y, grid))
        progress = s / total_s
        states.append(
            VehicleState(
                x=x,
                y=y,
                yaw=yaw,
                kappa=kappa,
                s=s,
                is_turn=is_turn,
                risk=r,
                progress=progress,
            )
        )
        risks.append(r)
        if is_turn:
            turn_risks.append(r)

    duration = float(times[-1]) if len(times) else 0.0
    return SimulationResult(
        states=states,
        mean_risk=float(np.mean(risks)) if risks else 0.0,
        max_risk=float(np.max(risks)) if risks else 0.0,
        mean_risk_turn=float(np.mean(turn_risks)) if turn_risks else 0.0,
        duration_s=duration,
    )
