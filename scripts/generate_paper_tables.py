#!/usr/bin/env python3
"""Generate RA-L/T-ASE style tables and figures from experiment outputs."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import AppConfig, load_config
from src.visualization import (
    plot_ablation_summary,
    plot_budget_experiment,
    plot_delta_sweep,
    plot_fallback_grouping,
    plot_pareto,
    plot_risk_decomposition,
    plot_runtime_comparison,
)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_subset(rows: list[dict], group_type: str, group_value: str) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "group_type": group_type,
        "group_value": group_value,
        "n_fields": n,
        "fallback_rate": sum(1 for r in rows if r["fallback"]) / n,
        "mean_path_length_m": sum(r["path_length_m"] for r in rows) / n,
        "mean_risk": sum(r["mean_risk"] for r in rows) / n,
        "mean_violation": sum(r["violation"] for r in rows) / n,
    }


def _coerce(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                **r,
                "path_length_m": float(r.get("path_length_m", 0)),
                "compaction_cost": float(r.get("compaction_cost", 0)),
                "mean_risk": float(r.get("mean_risk", 0)),
                "max_risk": float(r.get("max_risk", 0)),
                "coverage_rate": float(r.get("coverage_rate", 0)),
                "fallback": str(r.get("fallback", "")).lower() in ("true", "1"),
                "violation": float(r.get("violation", 0)),
                "delta": float(r.get("delta", 0.38)),
                "headland_cost": float(r.get("headland_cost", 0)),
                "hotspot_cost": float(r.get("hotspot_cost", 0)),
                "pass_count_cost": float(r.get("pass_count_cost", 0)),
                "repeat_cost": float(r.get("repeat_cost", 0)),
                "turn_cost": float(r.get("turn_cost", 0)),
                "runtime_full_assess_s": float(r.get("runtime_full_assess_s", 0)),
                "runtime_nr_pool_s": float(r.get("runtime_nr_pool_s", 0)),
                "field_area_m2": float(r.get("field_area_m2", 0)),
                "aspect_ratio": float(r.get("aspect_ratio", 0)),
                "boundary_complexity": float(r.get("boundary_complexity", 0)),
                "num_candidates": int(float(r.get("num_candidates", 0))),
                "num_informed_candidates": int(float(r.get("num_informed_candidates", 0))),
            }
        )
    return out


def _country_prefix(field_name: str) -> str:
    return field_name.split("_", 1)[0]


def _area_bin(area: float) -> str:
    if area < 5e4:
        return "<50k"
    if area < 1e5:
        return "50-100k"
    if area < 2e5:
        return "100-200k"
    return ">=200k"


def _aspect_bin(ar: float) -> str:
    if ar < 1.5:
        return "<1.5"
    if ar < 2.5:
        return "1.5-2.5"
    if ar < 4.0:
        return "2.5-4.0"
    return ">=4.0"


def generate_paper_tables(project_root: Path, config: AppConfig) -> int:
    results_dir = project_root / config.output.results_dir
    figures_dir = project_root / config.output.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    batch = _coerce(_read_csv(results_dir / "batch_results.csv"))
    if not batch:
        print("Missing batch_results.csv — run: python main.py batch")
        return 1

    # --- Method summary table ---
    method_rows = []
    for method in sorted(set(r["method"] for r in batch)):
        subset = [r for r in batch if r["method"] == method]
        n = len(subset)
        method_rows.append(
            {
                "method": method,
                "delta": config.selection.delta,
                "n_fields": n,
                "mean_path_length_m": sum(r["path_length_m"] for r in subset) / n,
                "mean_compaction_cost": sum(r["compaction_cost"] for r in subset) / n,
                "mean_risk": sum(r["mean_risk"] for r in subset) / n,
                "fallback_rate": sum(1 for r in subset if r["fallback"]) / n,
                "mean_violation": sum(r["violation"] for r in subset) / n,
                "mean_runtime_full_s": sum(r["runtime_full_assess_s"] for r in subset) / n,
                "mean_nr_pool_size": sum(r["num_informed_candidates"] for r in subset) / n,
            }
        )
    _write_csv(method_rows, results_dir / "paper_method_summary.csv")
    print(f"  saved paper_method_summary.csv")

    # --- Fallback grouping ---
    rb_rows = [r for r in batch if r["method"] == "rb_ccp"]
    nr_rows = [r for r in batch if r["method"] == "nr_ccp"]
    grouping_rows = []

    for prefix in sorted(set(_country_prefix(r["field_name"]) for r in rb_rows)):
        for method, subset_all in (("rb_ccp", rb_rows), ("nr_ccp", nr_rows)):
            subset = [r for r in subset_all if _country_prefix(r["field_name"]) == prefix]
            row = _summarize_subset(subset, "country", prefix)
            if row:
                row["method"] = method
                grouping_rows.append(row)

    for method, subset_all in (("rb_ccp", rb_rows), ("nr_ccp", nr_rows)):
        bins: dict[str, list] = defaultdict(list)
        for r in subset_all:
            bins[_area_bin(r["field_area_m2"])].append(r)
        for b, subset in sorted(bins.items()):
            row = _summarize_subset(subset, "area_bin", b)
            if row:
                row["method"] = method
                grouping_rows.append(row)

        ar_bins: dict[str, list] = defaultdict(list)
        for r in subset_all:
            ar_bins[_aspect_bin(r["aspect_ratio"])].append(r)
        for b, subset in sorted(ar_bins.items()):
            row = _summarize_subset(subset, "aspect_bin", b)
            if row:
                row["method"] = method
                grouping_rows.append(row)

    _write_csv(grouping_rows, results_dir / "paper_fallback_grouping.csv")
    print(f"  saved paper_fallback_grouping.csv")

    # --- Runtime comparison ---
    runtime_rows = []
    seen = set()
    for r in rb_rows:
        fn = r["field_name"]
        if fn in seen:
            continue
        seen.add(fn)
        nr = next((x for x in nr_rows if x["field_name"] == fn), None)
        runtime_rows.append(
            {
                "field_name": fn,
                "num_full_candidates": r["num_candidates"],
                "num_nr_candidates": nr["num_informed_candidates"] if nr else 0,
                "runtime_full_assess_s": r["runtime_full_assess_s"],
                "runtime_nr_pool_s": r["runtime_nr_pool_s"],
                "nr_pool_fraction": (
                    (nr["num_informed_candidates"] / r["num_candidates"])
                    if nr and r["num_candidates"]
                    else 0.0
                ),
            }
        )
    _write_csv(runtime_rows, results_dir / "paper_runtime_comparison.csv")
    print(f"  saved paper_runtime_comparison.csv")

    # --- Risk decomposition ---
    risk_rows = []
    for method in ("rb_ccp", "nr_ccp", "naive"):
        subset = [r for r in batch if r["method"] == method]
        if not subset:
            continue
        n = len(subset)
        risk_rows.append(
            {
                "method": method,
                "mean_headland_cost": sum(r["headland_cost"] for r in subset) / n,
                "mean_hotspot_cost": sum(r["hotspot_cost"] for r in subset) / n,
                "mean_pass_count_cost": sum(r["pass_count_cost"] for r in subset) / n,
                "mean_repeat_cost": sum(r["repeat_cost"] for r in subset) / n,
                "mean_turn_cost": sum(r["turn_cost"] for r in subset) / n,
                "mean_total_risk": sum(r["mean_risk"] for r in subset) / n,
            }
        )
    _write_csv(risk_rows, results_dir / "paper_risk_decomposition.csv")
    print(f"  saved paper_risk_decomposition.csv")

    # --- Figures ---
    plot_pareto(batch, figures_dir / "paper_pareto.png", dpi=config.output.dpi)
    plot_fallback_grouping(grouping_rows, figures_dir / "paper_fallback_grouping.png", dpi=config.output.dpi)
    plot_runtime_comparison(runtime_rows, figures_dir / "paper_runtime.png", dpi=config.output.dpi)
    plot_risk_decomposition(risk_rows, figures_dir / "paper_risk_decomposition.png", dpi=config.output.dpi)
    print(f"  saved paper figures in {figures_dir.relative_to(project_root)}")

    delta_rows = _coerce(_read_csv(results_dir / "delta_sweep_results.csv"))
    if delta_rows:
        plot_delta_sweep(delta_rows, figures_dir / "delta_sweep.png", dpi=config.output.dpi)
        print(f"  saved delta_sweep.png")

    ablation_rows = _read_csv(results_dir / "ablation_results.csv")
    if ablation_rows:
        plot_ablation_summary(ablation_rows, figures_dir / "paper_ablation.png", dpi=config.output.dpi)
        print(f"  saved paper_ablation.png")

    budget_rows = _read_csv(results_dir / "budget_experiment_results.csv")
    if budget_rows:
        plot_budget_experiment(
            budget_rows,
            figures_dir / "paper_budget_experiment.png",
            dpi=config.output.dpi,
        )
        print(f"  saved paper_budget_experiment.png")

    return 0


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "nr_ccp_full.yaml")
    raise SystemExit(generate_paper_tables(ROOT, cfg))
