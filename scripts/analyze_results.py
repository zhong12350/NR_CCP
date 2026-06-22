#!/usr/bin/env python3
"""Analyze batch_results.csv → Pareto, summary plots, and tables."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.visualization import plot_batch_summary, plot_delta_violation, plot_pareto


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python main.py batch")
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _coerce_batch_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                **r,
                "path_length_m": float(r["path_length_m"]),
                "compaction_cost": float(r["compaction_cost"]),
                "mean_risk": float(r["mean_risk"]),
                "max_risk": float(r["max_risk"]),
                "coverage_rate": float(r["coverage_rate"]),
                "fallback": r["fallback"].lower() in ("true", "1"),
                "violation": float(r["violation"]),
            }
        )
    return out


def _coerce_summary_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                **r,
                "mean_path_length_m": float(r["mean_path_length_m"]),
                "mean_compaction_cost": float(r["mean_compaction_cost"]),
                "mean_risk": float(r["mean_risk"]),
                "fallback_rate": float(r["fallback_rate"]),
                "delta": float(r.get("delta", 0.38)),
            }
        )
    return out


def main(project_root: Path | None = None, config_path: Path | None = None) -> int:
    root = project_root or ROOT
    cfg_path = config_path or (root / "configs" / "nr_ccp_full.yaml")
    config = load_config(cfg_path)
    results_dir = root / config.output.results_dir
    figures_dir = root / config.output.figures_dir

    batch_path = results_dir / "batch_results.csv"
    rows = _coerce_batch_rows(_read_csv(batch_path))

    plot_pareto(rows, figures_dir / "pareto.png", dpi=config.output.dpi)
    print(f"  saved {(figures_dir / 'pareto.png').relative_to(root)}")

    method_path = results_dir / "method_summary.csv"
    if method_path.exists():
        summary = _coerce_summary_rows(_read_csv(method_path))
        plot_batch_summary(summary, figures_dir / "batch_summary.png", dpi=config.output.dpi)
        print(f"  saved {(figures_dir / 'batch_summary.png').relative_to(root)}")

    violation_path = results_dir / "violation_summary.csv"
    if violation_path.exists():
        viol = _read_csv(violation_path)
        for r in viol:
            r["fallback_rate"] = float(r["fallback_rate"])
        plot_delta_violation(viol, figures_dir / "violation_rate.png", dpi=config.output.dpi)
        print(f"  saved {(figures_dir / 'violation_rate.png').relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
