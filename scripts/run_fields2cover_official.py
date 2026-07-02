#!/usr/bin/env python3
"""Run official Fields2Cover on Fields2Benchmark WKT files and export baseline CSV."""

from __future__ import annotations

import csv
import math
import os
import sys
import traceback
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import fields2cover as f2c
except ImportError as exc:
    raise SystemExit(
        "fields2cover not found. Inside Docker run:\n"
        "  source /workspace/scripts/docker/f2c_env.sh\n"
        f"Original error: {exc}"
    ) from exc

from src.config_loader import AppConfig, load_config


def _read_wkt(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("empty WKT")
    return text


def _path_length_m(path) -> float:
    total = 0.0
    for i in range(path.size()):
        state = path.getState(i)
        total += float(state.length)
    return total


def _swath_angle_deg(swaths) -> float:
    """Extract principal swath angle in degrees."""
    if hasattr(swaths, "getSwath"):
        swath = swaths.getSwath(0)
    elif hasattr(swaths, "getGeometry"):
        swath = swaths.getGeometry(0)
    else:
        swath = swaths

    if hasattr(swath, "getAngle"):
        angle_rad = float(swath.getAngle())
    elif hasattr(swath, "getPath"):
        line = swath.getPath()
        if hasattr(line, "getAngle"):
            angle_rad = float(line.getAngle())
        else:
            start = line.startPoint()
            end = line.endPoint()
            angle_rad = math.atan2(end.getY() - start.getY(), end.getX() - start.getX())
    else:
        raise RuntimeError("cannot extract swath angle from Fields2Cover output")

    angle_deg = math.degrees(angle_rad) % 180.0
    return round(angle_deg, 4)


def run_fields2cover_on_wkt(
    wkt_path: Path,
    swath_width_m: float = 6.0,
    headland_width_m: float = 18.0,
    min_turn_radius_m: float = 2.0,
) -> dict:
    """Official F2C complete flow for one field."""
    wkt_text = _read_wkt(wkt_path)

    cell = f2c.Cell()
    cell.importFromWkt(wkt_text)
    cells = f2c.Cells()
    cells.addGeometry(cell)
    field = f2c.Field()
    field.setField(cells)
    f2c.Transform.transformToUTM(field)

    robot = f2c.Robot(2.0, swath_width_m)
    robot.setMinTurningRadius(min_turn_radius_m)

    const_hl = f2c.HG_Const_gen()
    no_hl = const_hl.generateHeadlands(field.getField(), headland_width_m)

    bf = f2c.SG_BruteForce()
    swaths = bf.generateSwaths(math.pi, robot.getCovWidth(), no_hl.getGeometry(0))

    snake = f2c.RP_Snake()
    swaths = snake.genSortedSwaths(swaths)

    angle_deg = _swath_angle_deg(swaths)

    path_planner = f2c.PP_PathPlanning()
    dubins = f2c.PP_DubinsCurves()
    path = path_planner.planPath(robot, swaths, dubins)
    path_length_m = _path_length_m(path)

    return {
        "field_name": wkt_path.stem,
        "angle_deg": angle_deg,
        "path_length_m": path_length_m,
        "source": "fields2cover_official",
    }


def _max_fields(config: AppConfig) -> int:
    raw = os.getenv("NR_CCP_F2C_FIELDS")
    if raw:
        return min(config.batch.max_fields, int(raw))
    return config.batch.max_fields


def run_fields2cover_official(config: AppConfig, project_root: Path) -> int:
    pattern = str(project_root / config.batch.field_glob)
    wkt_files = sorted(glob(pattern))[: _max_fields(config)]
    out_csv = project_root / "data" / "fields2cover" / "official_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failed: list[str] = []
    print(
        f"Official Fields2Cover on {len(wkt_files)} fields | "
        f"swath={config.planner.swath_width_m}m headland={config.headland.width_m}m"
    )

    for i, wkt in enumerate(wkt_files, start=1):
        name = Path(wkt).stem
        if i == 1 or i % 25 == 0 or i == len(wkt_files):
            print(f"  [{i}/{len(wkt_files)}] {name}")
        try:
            row = run_fields2cover_on_wkt(
                Path(wkt),
                swath_width_m=config.planner.swath_width_m,
                headland_width_m=config.headland.width_m,
            )
            rows.append(row)
        except Exception as exc:
            failed.append(name)
            print(f"    FAIL {name}: {exc}")
            if os.getenv("NR_CCP_F2C_VERBOSE") == "1":
                traceback.print_exc()

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["field_name", "angle_deg", "path_length_m", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  saved {out_csv} ({len(rows)} ok, {len(failed)} failed)")
    if failed:
        print(f"  failed fields: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    return 0 if rows else 1


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "nr_ccp_full.yaml")
    raise SystemExit(run_fields2cover_official(cfg, ROOT))
