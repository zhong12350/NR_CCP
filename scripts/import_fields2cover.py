#!/usr/bin/env python3
"""Import or generate Fields2Cover baseline CSV for fair comparison."""

from __future__ import annotations

import csv
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import AppConfig, load_config
from src.fields2cover_baseline import (
    Fields2CoverRecord,
    _choose_f2c_angle,
    load_official_csv,
    try_run_fields2cover_cli,
)
from src.geometry import load_field_from_wkt
from src.assessor import path_length
from src.candidates import generate_candidate


def import_fields2cover(config: AppConfig, project_root: Path) -> int:
    out_dir = project_root / "data" / "fields2cover"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "official_results.csv"

    existing = load_official_csv(out_csv)
    pattern = str(project_root / config.batch.field_glob)
    wkt_files = sorted(glob(pattern))[: config.batch.max_fields]

    rows: list[dict] = []
    print(f"Generating Fields2Cover baseline for {len(wkt_files)} fields → {out_csv}")

    for i, wkt in enumerate(wkt_files, start=1):
        name = Path(wkt).stem
        if name in existing and existing[name].source == "official_csv":
            rec = existing[name]
            rows.append(
                {
                    "field_name": name,
                    "angle_deg": rec.angle_deg,
                    "path_length_m": rec.path_length_m,
                    "source": rec.source,
                }
            )
            continue

        if i == 1 or i % 25 == 0 or i == len(wkt_files):
            print(f"  [{i}/{len(wkt_files)}] {name}")

        source = "f2c_heuristic"
        angle = 0.0
        length = 0.0
        try:
            tmp_csv = out_dir / f"{name}_f2c.csv"
            if try_run_fields2cover_cli(Path(wkt), tmp_csv):
                with tmp_csv.open("r", encoding="utf-8") as f:
                    row = next(csv.DictReader(f), None)
                if row:
                    angle = float(row.get("angle_deg", 0.0))
                    length = float(row.get("path_length_m", 0.0))
                    source = "fields2cover_cli"
            else:
                grid = load_field_from_wkt(
                    wkt,
                    config.headland.width_m,
                    auto_cell_size=config.field.auto_cell_size,
                )
                angle = _choose_f2c_angle(grid, config.planner.swath_width_m)
                cand = generate_candidate(grid, angle, config.planner)
                length = path_length(cand.waypoints) if cand else 0.0
        except Exception as exc:
            print(f"    skip {name}: {exc}")
            continue

        rows.append(
            {
                "field_name": name,
                "angle_deg": angle,
                "path_length_m": length,
                "source": source,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["field_name", "angle_deg", "path_length_m", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)

    sources = {}
    for r in rows:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    print(f"  saved {len(rows)} records | sources: {sources}")
    return 0


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "nr_ccp_full.yaml")
    raise SystemExit(import_fields2cover(cfg, ROOT))
