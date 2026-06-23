"""Matplotlib animation for coverage-path simulation."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Polygon

from src.fields import RiskField
from src.geometry import FieldGrid
from src.simulation import SimulationResult, VehicleState
from src.trajectory import ExecutableMetrics, ExecutableTrajectory, VehicleMotionConfig
from src.visualization import _label


def _draw_polygon(ax, poly: Polygon, **kwargs) -> None:
    x, y = poly.exterior.xy
    ax.add_patch(MplPolygon(np.column_stack([x, y]), closed=True, **kwargs))


def _vehicle_body_corners(
    x: float, y: float, yaw: float, length: float, width: float
) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([length / 2, length / 2, -length / 2, -length / 2])
    dy = np.array([width / 2, -width / 2, -width / 2, width / 2])
    xs = x + c * dx - s * dy
    ys = y + s * dx + c * dy
    return np.column_stack([xs, ys])


def _swath_corners(
    x: float, y: float, yaw: float, length: float, swath_width: float
) -> np.ndarray:
    """Implement footprint centered on vehicle."""
    c, s = np.cos(yaw), np.sin(yaw)
    half_w = swath_width / 2.0
    dx = np.array([length / 2, length / 2, -length / 2, -length / 2])
    dy = np.array([half_w, -half_w, -half_w, half_w])
    xs = x + c * dx - s * dy
    ys = y + s * dx + c * dy
    return np.column_stack([xs, ys])


def _risk_color(r: float) -> tuple[float, float, float, float]:
    """Green (low) → yellow → red (high)."""
    r = float(np.clip(r, 0.0, 1.0))
    if r < 0.5:
        t = r / 0.5
        return (0.2 + 0.6 * t, 0.85 - 0.35 * t, 0.25, 0.95)
    t = (r - 0.5) / 0.5
    return (0.8 + 0.2 * t, 0.5 - 0.45 * t, 0.1, 0.95)


def _pick_writer(out_path: Path, fps: int):
    suffix = out_path.suffix.lower()
    if suffix == ".mp4":
        if shutil.which("ffmpeg"):
            return FFMpegWriter(fps=fps, bitrate=4000, metadata={"artist": "NR-CCP"})
        return None
    if suffix == ".gif":
        return PillowWriter(fps=fps)
    return PillowWriter(fps=fps)


def animate_coverage_simulation(
    grid: FieldGrid,
    risk: RiskField,
    traj: ExecutableTrajectory,
    sim: SimulationResult,
    motion: VehicleMotionConfig,
    exec_metrics: ExecutableMetrics,
    method: str,
    out_path: Path,
    field_name: str = "field",
    mean_risk_geo: float | None = None,
    fps: int = 24,
    trail_window: int = 180,
    dpi: int = 120,
) -> Path:
    """Render a polished vehicle-following animation and save to GIF or MP4."""
    states = sim.states
    if not states:
        raise ValueError("Simulation produced no states")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() not in (".gif", ".mp4"):
        out_path = out_path.with_suffix(".gif")

    # Subsample frames for very long paths while keeping smooth playback.
    target_frames = min(len(states), int(fps * 90))
    if len(states) > target_frames:
        idx = np.linspace(0, len(states) - 1, target_frames, dtype=int)
        frame_states = [states[i] for i in idx]
    else:
        frame_states = states

    fig = plt.figure(figsize=(12, 8), facecolor="#0f1419")
    ax = fig.add_axes([0.06, 0.08, 0.68, 0.86])
    hud_ax = fig.add_axes([0.76, 0.08, 0.22, 0.86])
    hud_ax.set_facecolor("#1a2332")
    hud_ax.axis("off")

    minx, miny, maxx, maxy = grid.geometry.bounds
    extent = [minx, maxx, miny, maxy]
    ax.set_facecolor("#1e2a22")

    base = np.ma.masked_where(~grid.outer_mask, risk.values)
    ax.imshow(
        base,
        origin="lower",
        extent=extent,
        cmap="inferno",
        alpha=0.82,
        vmin=0,
        vmax=1,
        aspect="equal",
        zorder=0,
    )

    _draw_polygon(
        ax,
        grid.geometry.inner,
        fill=False,
        edgecolor="#e8f4ea",
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
        zorder=2,
    )
    _draw_polygon(
        ax,
        grid.geometry.outer,
        fill=False,
        edgecolor="white",
        linewidth=1.8,
        zorder=3,
    )

    geo_x = [p[0] for p in traj.geo_waypoints]
    geo_y = [p[1] for p in traj.geo_waypoints]
    ax.plot(geo_x, geo_y, color="#8899aa", linewidth=1.0, linestyle=":", alpha=0.55, zorder=4)

    exec_line, = ax.plot(
        traj.xs,
        traj.ys,
        color="#4da3ff",
        linewidth=1.2,
        alpha=0.35,
        linestyle="--",
        zorder=5,
    )
    turn_mask = traj.is_turn
    if np.any(turn_mask):
        ax.scatter(
            traj.xs[turn_mask],
            traj.ys[turn_mask],
            s=8,
            c="#ff6b6b",
            alpha=0.45,
            zorder=6,
            label="Turn segments",
        )

    trail_lc = LineCollection([], linewidths=3.0, capstyle="round", zorder=8)
    ax.add_collection(trail_lc)

    swath_patch = MplPolygon([[0, 0]] * 4, closed=True, facecolor="#7cfc00", edgecolor="none", alpha=0.22, zorder=9)
    ax.add_patch(swath_patch)

    body_patch = MplPolygon([[0, 0]] * 4, closed=True, facecolor="#39ff14", edgecolor="white", linewidth=1.5, alpha=0.95, zorder=11)
    ax.add_patch(body_patch)

    heading_arrow = FancyArrowPatch(
        (0, 0),
        (1, 0),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=2.0,
        color="white",
        zorder=12,
    )
    ax.add_patch(heading_arrow)

    glow = plt.Circle((0, 0), radius=1.0, facecolor="#ff4444", edgecolor="none", alpha=0.0, zorder=10)
    ax.add_patch(glow)

    ax.scatter(states[0].x, states[0].y, c="#7fff00", s=90, edgecolors="white", linewidths=1.2, zorder=13, label="Start")
    ax.scatter(states[-1].x, states[-1].y, c="white", s=55, edgecolors="black", linewidths=1.0, zorder=13, label="Goal")

    ax.set_xlim(minx - 5, maxx + 5)
    ax.set_ylim(miny - 5, maxy + 5)
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.tick_params(colors="#c8d6e5")
    for spine in ax.spines.values():
        spine.set_color("#5a6a7a")

    title = (
        f"{field_name}  |  {_label(method)}  |  Executable coverage simulation\n"
        f"L_exec={exec_metrics.path_length_exec_m:.0f} m  "
        f"κ_max={exec_metrics.max_curvature:.3f}  "
        f"turns={exec_metrics.num_turns}"
    )
    fig.suptitle(title, color="white", fontsize=11, y=0.98)

    hud_text = hud_ax.text(
        0.05,
        0.97,
        "",
        transform=hud_ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#ecf0f1",
        family="monospace",
        linespacing=1.45,
    )

    progress_bg = hud_ax.add_patch(
        plt.Rectangle((0.05, 0.04), 0.9, 0.025, transform=hud_ax.transAxes, facecolor="#2c3e50", edgecolor="none")
    )
    progress_fill = hud_ax.add_patch(
        plt.Rectangle((0.05, 0.04), 0.0, 0.025, transform=hud_ax.transAxes, facecolor="#2ecc71", edgecolor="none")
    )

    risk_bar_ax = fig.add_axes([0.06, 0.02, 0.68, 0.025])
    risk_bar_ax.set_facecolor("#1a2332")
    risk_bar_ax.set_xlim(0, 1)
    risk_bar_ax.set_ylim(0, 1)
    risk_bar_ax.axis("off")
    risk_bar = risk_bar_ax.barh([0.5], [0.0], height=0.8, color="#2ecc71", edgecolor="none")[0]
    risk_bar_ax.text(0.5, 0.5, "Risk at vehicle", ha="center", va="center", color="#bdc3c7", fontsize=8)

    def _update_hud(st: VehicleState, frame: int, mean_risk_so_far: float) -> None:
        geo_risk = f"{mean_risk_geo:.3f}" if mean_risk_geo is not None else "n/a"
        mode = "TURN" if st.is_turn else "SWATH"
        hud_text.set_text(
            f"FRAME  {frame+1}/{len(frame_states)}\n"
            f"TIME   {st.progress * sim.duration_s:.1f} s\n"
            f"DIST   {st.s:.0f} / {exec_metrics.path_length_exec_m:.0f} m\n"
            f"SPEED  {motion.cruise_speed_mps:.1f} m/s\n"
            f"MODE   {mode}\n"
            f"YAW    {np.degrees(st.yaw):+.0f} deg\n"
            f"KAPPA  {st.kappa:+.3f} 1/m\n"
            f"RISK   {st.risk:.3f}\n"
            f"MEAN   {mean_risk_so_far:.3f}\n"
            f"GEO μR {geo_risk}\n"
            f"TURN μR {sim.mean_risk_turn:.3f}\n"
            f"ΔL     +{exec_metrics.length_increase_pct:.1f}%"
        )
        progress_fill.set_width(0.9 * st.progress)
        risk_bar.set_width(st.risk)
        risk_bar.set_color(plt.cm.inferno(st.risk))

    def update(frame: int):
        st = frame_states[frame]
        hist = frame_states[max(0, frame - trail_window) : frame + 1]
        if len(hist) >= 2:
            segs = [
                [(h.x, h.y), (hist[i + 1].x, hist[i + 1].y)]
                for i, h in enumerate(hist[:-1])
            ]
            colors = [_risk_color(h.risk) for h in hist[:-1]]
            trail_lc.set_segments(segs)
            trail_lc.set_color(colors)

        body_patch.set_xy(_vehicle_body_corners(st.x, st.y, st.yaw, motion.body_length_m, motion.body_width_m))
        swath_patch.set_xy(_swath_corners(st.x, st.y, st.yaw, motion.body_length_m * 1.1, motion.swath_width_m))

        tip = 1.2
        heading_arrow.set_positions(
            (st.x, st.y),
            (st.x + tip * np.cos(st.yaw), st.y + tip * np.sin(st.yaw)),
        )

        if st.is_turn:
            glow.center = (st.x, st.y)
            glow.set_radius(motion.min_turn_radius_m * 0.35)
            glow.set_alpha(0.25)
        else:
            glow.set_alpha(0.0)

        mean_risk = float(np.mean([s.risk for s in frame_states[: frame + 1]]))
        _update_hud(st, frame, mean_risk)

        artists = [trail_lc, body_patch, swath_patch, heading_arrow, glow, hud_text, progress_fill, risk_bar]
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_states),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    writer = _pick_writer(out_path, fps)
    saved_path = out_path
    if writer is None and out_path.suffix.lower() == ".mp4":
        saved_path = out_path.with_suffix(".gif")

    writer = writer or PillowWriter(fps=fps)
    print(f"  rendering {len(frame_states)} frames → {saved_path.name} ({fps} fps)...")
    anim.save(str(saved_path), writer=writer, dpi=dpi)
    plt.close(fig)

    if saved_path.suffix.lower() == ".mp4" and saved_path.stat().st_size < 1024:
        gif_path = saved_path.with_suffix(".gif")
        anim.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=dpi)
        saved_path = gif_path

    return saved_path


def animate_method_comparison(
    grid: FieldGrid,
    risk: RiskField,
    left: tuple[str, ExecutableTrajectory, SimulationResult, ExecutableMetrics],
    right: tuple[str, ExecutableTrajectory, SimulationResult, ExecutableMetrics],
    motion: VehicleMotionConfig,
    out_path: Path,
    field_name: str = "field",
    fps: int = 20,
    dpi: int = 110,
) -> Path:
    """Side-by-side comparison animation (e.g. naive vs rb_ccp)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() not in (".gif", ".mp4"):
        out_path = out_path.with_suffix(".gif")

    panels = [left, right]
    frame_sets = []
    for _, traj, sim, _ in panels:
        states = sim.states
        target = min(len(states), int(fps * 60))
        if len(states) > target:
            idx = np.linspace(0, len(states) - 1, target, dtype=int)
            frame_sets.append([states[i] for i in idx])
        else:
            frame_sets.append(states)
    n_frames = min(len(fs) for fs in frame_sets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f1419")
    artists_all: list = []

    for ax, (method, traj, sim, metrics), frame_states in zip(axes, panels, frame_sets):
        ax.set_facecolor("#1e2a22")
        minx, miny, maxx, maxy = grid.geometry.bounds
        base = np.ma.masked_where(~grid.outer_mask, risk.values)
        ax.imshow(base, origin="lower", extent=[minx, maxx, miny, maxy], cmap="inferno", alpha=0.75, vmin=0, vmax=1, aspect="equal")
        _draw_polygon(ax, grid.geometry.outer, fill=False, edgecolor="white", linewidth=1.2)
        ax.plot(traj.xs, traj.ys, color="#4da3ff", lw=0.8, alpha=0.3, ls="--")

        trail, = ax.plot([], [], color="#39ff14", lw=2.2, solid_capstyle="round")
        body = MplPolygon([[0, 0]] * 4, closed=True, facecolor="#39ff14", edgecolor="white", lw=1.2, alpha=0.9)
        ax.add_patch(body)
        ax.set_xlim(minx - 3, maxx + 3)
        ax.set_ylim(miny - 3, maxy + 3)
        ax.set_title(
            f"{_label(method)}\nL={metrics.path_length_exec_m:.0f}m  μR={sim.mean_risk:.3f}",
            color="white",
            fontsize=10,
        )
        ax.tick_params(colors="#aaa")
        artists_all.append((trail, body, frame_states))

    fig.suptitle(f"{field_name} — Method comparison", color="white", fontsize=12)

    def update(frame: int):
        out = []
        for ax_idx, (trail, body, frame_states) in enumerate(artists_all):
            st = frame_states[min(frame, len(frame_states) - 1)]
            hist = frame_states[: frame + 1]
            trail.set_data([s.x for s in hist], [s.y for s in hist])
            body.set_xy(_vehicle_body_corners(st.x, st.y, st.yaw, motion.body_length_m, motion.body_width_m))
            out.extend([trail, body])
        return out

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    writer = _pick_writer(out_path, fps) or PillowWriter(fps=fps)
    saved = out_path if _pick_writer(out_path, fps) else out_path.with_suffix(".gif")
    if saved.suffix == ".mp4" and not shutil.which("ffmpeg"):
        saved = out_path.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
    anim.save(str(saved), writer=writer, dpi=dpi)
    plt.close(fig)
    return saved
