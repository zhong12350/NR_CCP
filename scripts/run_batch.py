#!/usr/bin/env python3
"""Batch sweep over strip widths (optional extended experiments)."""

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from main import run_pipeline
from src.config_loader import load_config


def main() -> int:
    root = ROOT
    base = load_config(root / "configs" / "default.yaml")
    strip_widths = [3.0, 4.0, 5.0, 6.0]

    all_rows = []
    for w in strip_widths:
        cfg = deepcopy(base)
        cfg.planner.strip_width_m = w
        cfg.output.figures_dir = Path(f"outputs/batch/strip_{w:.0f}m/figures")
        cfg.output.results_dir = Path(f"outputs/batch/strip_{w:.0f}m/results")
        print(f"\n=== strip_width = {w} m ===")
        _, metrics = run_pipeline(cfg, root)
        for m in metrics:
            row = m.to_dict()
            row["strip_width_m"] = w
            all_rows.append(row)

    out = root / "outputs" / "batch" / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        import csv

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nBatch summary: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
