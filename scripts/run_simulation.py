#!/usr/bin/env python3
"""Plan a field and render an executable-path Matplotlib animation."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.animation import animate_coverage_simulation, animate_method_comparison
from src.config_loader import AppConfig, load_config
from src.metrics import metrics_from_selection
from src.planner import plan_field
from src.simulation import simulate_along_trajectory
from src.trajectory import (
    VehicleMotionConfig,
    compute_executable_metrics,
    postprocess_executable,
)


def _motion_from_config(config: AppConfig, args) -> VehicleMotionConfig:
    return VehicleMotionConfig(
        min_turn_radius_m=args.turn_radius,
        swath_width_m=config.planner.swath_width_m,
        cruise_speed_mps=args.speed,
    )


def _selection_by_method(result, method: str):
    for sel in result.selections:
        if sel.method == method:
            return sel
    raise ValueError(
        f"Method '{method}' not in run results. Available: {[s.method for s in result.selections]}"
    )


def _write_sim_csv(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_simulation(
    config: AppConfig,
    project_root: Path,
    wkt_path: Path | None = None,
    method: str = "rb_ccp",
    compare: list[str] | None = None,
    out_path: Path | None = None,
    turn_radius: float = 5.0,
    speed: float = 2.0,
    fps: int = 24,
    format: str = "auto",
) -> int:
    wkt = wkt_path or (project_root / config.field.wkt_path)
    sim_dir = project_root / "outputs" / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== NR-CCP Simulation ===")
    print(f"  field   : {wkt.name}")
    print(f"  method  : {method}")
    print(f"  ρ_min   : {turn_radius} m")
    print(f"  speed   : {speed} m/s")
    print(f"  fps     : {fps}")

    t0 = time.perf_counter()
    result = plan_field(wkt, config, project_root=project_root)
    print(f"  planned in {time.perf_counter() - t0:.1f}s")

    motion = _motion_from_config(
        config,
        argparse.Namespace(turn_radius=turn_radius, speed=speed),
    )

    if compare and len(compare) >= 2:
        left_m, right_m = compare[0], compare[1]
        panels = []
        for m in (left_m, right_m):
            sel = _selection_by_method(result, m)
            geo_metrics = metrics_from_selection(
                result.field_name,
                sel,
                len(result.full_assessments),
                len(result.informed_assessments),
                config.selection.delta,
                certificate=result.certificate,
            )
            traj = postprocess_executable(sel.assessment.candidate.waypoints, motion)
            exec_m = compute_executable_metrics(traj, motion)
            sim = simulate_along_trajectory(traj, result.risk, result.grid, motion)
            panels.append((m, traj, sim, exec_m))
            print(
                f"  {m}: L_exec={exec_m.path_length_exec_m:.0f}m, "
                f"κ_max={exec_m.max_curvature:.3f}, sim_μR={sim.mean_risk:.3f}"
            )

        dest = out_path or sim_dir / f"{result.field_name}_{left_m}_vs_{right_m}.gif"
        saved = animate_method_comparison(
            result.grid,
            result.risk,
            panels[0],
            panels[1],
            motion,
            dest,
            field_name=result.field_name,
            fps=fps,
        )
        print(f"  saved {saved.relative_to(project_root)}")
        return 0

    sel = _selection_by_method(result, method)
    geo_metrics = metrics_from_selection(
        result.field_name,
        sel,
        len(result.full_assessments),
        len(result.informed_assessments),
        config.selection.delta,
        certificate=result.certificate,
    )

    traj = postprocess_executable(sel.assessment.candidate.waypoints, motion)
    exec_metrics = compute_executable_metrics(traj, motion)
    sim = simulate_along_trajectory(traj, result.risk, result.grid, motion)

    ext = ".mp4" if format == "mp4" else ".gif"
    if format == "auto":
        import shutil

        ext = ".mp4" if shutil.which("ffmpeg") else ".gif"
    dest = out_path or sim_dir / f"{result.field_name}_{method}{ext}"

    saved = animate_coverage_simulation(
        result.grid,
        result.risk,
        traj,
        sim,
        motion,
        exec_metrics,
        method=method,
        out_path=dest,
        field_name=result.field_name,
        mean_risk_geo=geo_metrics.mean_risk,
        fps=fps,
    )

    csv_path = sim_dir / "simulation_metrics.csv"
    _write_sim_csv(
        {
            "field_name": result.field_name,
            "method": method,
            "path_length_geo_m": exec_metrics.path_length_geo_m,
            "path_length_exec_m": exec_metrics.path_length_exec_m,
            "length_increase_pct": exec_metrics.length_increase_pct,
            "num_turns": exec_metrics.num_turns,
            "max_curvature": exec_metrics.max_curvature,
            "turning_length_m": exec_metrics.turning_length_m,
            "mean_risk_geo": geo_metrics.mean_risk,
            "mean_risk_sim": sim.mean_risk,
            "mean_risk_turn": sim.mean_risk_turn,
            "max_risk_sim": sim.max_risk,
            "duration_s": sim.duration_s,
            "min_turn_radius_m": turn_radius,
            "cruise_speed_mps": speed,
            "animation": str(saved.name),
        },
        csv_path,
    )

    print(f"\n  Executable metrics:")
    print(f"    L_geo  = {exec_metrics.path_length_geo_m:.0f} m")
    print(f"    L_exec = {exec_metrics.path_length_exec_m:.0f} m (+{exec_metrics.length_increase_pct:.1f}%)")
    print(f"    κ_max  = {exec_metrics.max_curvature:.3f} 1/m  (limit {1/turn_radius:.3f})")
    print(f"    turns  = {exec_metrics.num_turns}, turn_len = {exec_metrics.turning_length_m:.0f} m")
    print(f"    sim μR = {sim.mean_risk:.3f}  (geo μR = {geo_metrics.mean_risk:.3f})")
    print(f"    turn μR= {sim.mean_risk_turn:.3f}")
    print(f"  saved {saved.relative_to(project_root)}")
    print(f"  saved {csv_path.relative_to(project_root)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NR-CCP coverage path animation")
    parser.add_argument("config", nargs="?", default="configs/default.yaml", help="YAML config")
    parser.add_argument("wkt", nargs="?", default=None, help="WKT field path")
    parser.add_argument("--method", default="rb_ccp", help="Selection method to animate")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="Side-by-side animation for two methods",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output GIF/MP4 path")
    parser.add_argument("--turn-radius", type=float, default=5.0, help="Min turn radius (m)")
    parser.add_argument("--speed", type=float, default=2.0, help="Cruise speed (m/s)")
    parser.add_argument("--fps", type=int, default=24, help="Animation frame rate")
    parser.add_argument(
        "--format",
        choices=("auto", "gif", "mp4"),
        default="auto",
        help="Output format (mp4 needs ffmpeg)",
    )
    args = parser.parse_args(argv)

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

    return run_simulation(
        config,
        root,
        wkt_path=wkt_path,
        method=args.method,
        compare=list(args.compare) if args.compare else None,
        out_path=args.out,
        turn_radius=args.turn_radius,
        speed=args.speed,
        fps=args.fps,
        format=args.format,
    )


if __name__ == "__main__":
    raise SystemExit(main())
