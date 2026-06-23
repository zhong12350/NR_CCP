#!/usr/bin/env python3
"""Δ sweep experiment: violation rate, path length, and risk vs Δ."""

from __future__ import annotations

import csv
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import AppConfig, load_config
from src.fields2cover_baseline import load_official_records
from src.metrics import metrics_from_selection
from src.planner import plan_field
from src.visualization import plot_delta_sweep


def run_delta_sweep(config: AppConfig, project_root: Path) -> int:
    deltas = [0.38, 0.50, 0.70, 0.85, 1.0]
    pattern = str(project_root / config.batch.field_glob)
    wkt_files = sorted(glob(pattern))[: min(config.batch.max_fields, 100)]
    official = load_official_records(project_root)

    rows: list[dict] = []
    print(f"Δ sweep on {len(wkt_files)} fields, Δ ∈ {deltas}")
    for delta in deltas:
        cfg = deepcopy(config)
        cfg.selection.delta = delta
        print(f"\n--- Δ = {delta} ---")
        for i, wkt in enumerate(wkt_files, start=1):
            if i == 1 or i % 25 == 0 or i == len(wkt_files):
                print(f"  [{i}/{len(wkt_files)}] {Path(wkt).stem}")
            try:
                result = plan_field(wkt, cfg, project_root=project_root, official_records=official)
                for sel in result.selections:
                    if sel.method not in ("rb_ccp", "nr_ccp", "naive", "fields2cover"):
                        continue
                    m = metrics_from_selection(
                        result.field_name,
                        sel,
                        len(result.full_assessments),
                        len(result.informed_assessments),
                        delta,
                        certificate=result.certificate,
                        runtime_full_assess_s=result.runtime_full_assess_s,
                        runtime_nr_pool_s=result.runtime_nr_pool_s,
                    )
                    rows.append(m.to_dict())
            except Exception as exc:
                print(f"  skip {Path(wkt).stem}: {exc}")

    out_dir = project_root / config.output.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "delta_sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n  saved {csv_path} ({len(rows)} rows)")

    fig_dir = project_root / config.output.figures_dir
    plot_delta_sweep(rows, fig_dir / "delta_sweep.png", dpi=config.output.dpi)
    print(f"  saved {fig_dir / 'delta_sweep.png'}")
    return 0


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "nr_ccp_full.yaml")
    raise SystemExit(run_delta_sweep(cfg, ROOT))
