#!/usr/bin/env python3
"""Plan a field and replay the selected path in MuJoCo."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import AppConfig, load_config
from src.mujoco_viz import MujocoReplayConfig, check_scene_loads, replay_in_viewer
from src.planner import plan_field
from src.simulation import simulate_along_trajectory
from src.trajectory import VehicleMotionConfig, postprocess_executable


def _motion_from_config(config: AppConfig, args) -> VehicleMotionConfig:
    return VehicleMotionConfig(
        min_turn_radius_m=args.turn_radius,
        swath_width_m=config.planner.swath_width_m,
        wheelbase_m=2.5,
        body_length_m=3.5,
        body_width_m=2.0,
        cruise_speed_mps=args.speed,
    )


def _ensure_mjpython_for_viewer() -> None:
    """MuJoCo's passive viewer on macOS must run under mjpython (Cocoa main thread)."""
    if sys.platform != "darwin":
        return
    if os.environ.get("MJPYTHON_BIN"):
        return

    mjpython = shutil.which("mjpython")
    if mjpython is None:
        raise RuntimeError(
            "macOS 上打开 MuJoCo 窗口需要用 mjpython，不能用普通 python。\n"
            "请改用:\n"
            "  mjpython main.py mujoco configs/default.yaml wkt/ee_field_10.wkt\n"
            "或:\n"
            "  mjpython scripts/run_mujoco_viz.py configs/default.yaml wkt/ee_field_10.wkt"
        )

    os.execvp(mjpython, [mjpython, *sys.argv])


def _selection_by_method(result, method: str):
    for sel in result.selections:
        if sel.method == method:
            return sel
    available = [s.method for s in result.selections]
    raise ValueError(f"Method '{method}' not in run results. Available: {available}")


def run_mujoco_viz(
    config: AppConfig,
    project_root: Path,
    wkt_path: Path | None = None,
    method: str = "rb_ccp",
    turn_radius: float = 5.0,
    speed: float = 2.0,
    playback_speed: float = 1.0,
    loop: bool = True,
    check_only: bool = False,
) -> int:
    wkt = wkt_path or (project_root / config.field.wkt_path)

    print("\n=== NR-CCP MuJoCo Visualization ===")
    print(f"  field   : {wkt.name}")
    print(f"  method  : {method}")
    print(f"  ρ_min   : {turn_radius} m")
    print(f"  speed   : {speed} m/s (simulation)")
    print(f"  playback: {playback_speed}x")

    t0 = time.perf_counter()
    result = plan_field(wkt, config, project_root=project_root)
    print(f"  planned in {time.perf_counter() - t0:.1f}s")

    motion = _motion_from_config(
        config,
        argparse.Namespace(turn_radius=turn_radius, speed=speed),
    )
    sel = _selection_by_method(result, method)
    traj = postprocess_executable(sel.assessment.candidate.waypoints, motion)
    sim = simulate_along_trajectory(traj, result.risk, result.grid, motion)

    replay_cfg = MujocoReplayConfig(speed=playback_speed, loop=loop)

    if check_only:
        info = check_scene_loads(result.grid, motion)
        print("\n  Scene check OK:")
        print(f"    bodies = {info['nbody']}")
        print(f"    geoms  = {info['ngeom']}")
        print(f"    center = {info['field_center_local']}")
        print(f"    states = {len(sim.states)}")
        return 0

    replay_in_viewer(
        result.grid,
        sim,
        motion,
        replay=replay_cfg,
        field_name=result.field_name,
        method=method,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NR-CCP MuJoCo 3D path replay")
    parser.add_argument("config", nargs="?", default="configs/default.yaml", help="YAML config")
    parser.add_argument("wkt", nargs="?", default=None, help="WKT field path")
    parser.add_argument("--method", default="rb_ccp", help="Selection method to visualize")
    parser.add_argument("--turn-radius", type=float, default=5.0, help="Min turn radius (m)")
    parser.add_argument("--speed", type=float, default=2.0, help="Sim cruise speed (m/s)")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Viewer replay speed multiplier",
    )
    parser.add_argument("--no-loop", action="store_true", help="Stop at end instead of looping")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compile scene without opening viewer (headless sanity check)",
    )
    args = parser.parse_args(argv)

    if not args.check:
        _ensure_mjpython_for_viewer()

    root = ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)

    wkt_path = None
    if args.wkt:
        wkt_path = Path(args.wkt)
        if not wkt_path.is_absolute():
            wkt_path = root / wkt_path

    return run_mujoco_viz(
        config,
        root,
        wkt_path=wkt_path,
        method=args.method,
        turn_radius=args.turn_radius,
        speed=args.speed,
        playback_speed=args.playback_speed,
        loop=not args.no_loop,
        check_only=args.check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
