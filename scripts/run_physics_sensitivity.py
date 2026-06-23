#!/usr/bin/env python3
"""Physics sensitivity: wheel load, tire pressure, soil moisture."""

from __future__ import annotations

import csv
import os
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import AppConfig, load_config
from src.metrics import metrics_from_selection
from src.physics import compute_physics_factors
from src.planner import plan_field


def _apply_override(cfg: AppConfig, key: str, value) -> None:
    parts = key.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _max_fields(config: AppConfig) -> int:
    raw = os.getenv("NR_CCP_PHYSICS_FIELDS")
    if raw:
        return min(config.batch.max_fields, int(raw))
    return min(config.batch.max_fields, 50)


SWEEPS = [
    ("wheel_load", "vehicle.mass_kg", [1500.0, 3000.0, 5000.0, 8000.0]),
    ("tire_pressure", "vehicle.contact_length_m", [0.30, 0.45, 0.60, 0.75]),
    ("soil_moisture", "soil.moisture", [0.15, 0.25, 0.35, 0.45]),
]


def run_physics_sensitivity(config: AppConfig, project_root: Path) -> int:
    pattern = str(project_root / config.batch.field_glob)
    wkt_files = sorted(glob(pattern))[: _max_fields(config)]
    rows: list[dict] = []

    print(f"Physics sensitivity on {len(wkt_files)} fields")

    for sweep_name, key, values in SWEEPS:
        print(f"\n--- sweep: {sweep_name} ---")
        for value in values:
            cfg = deepcopy(config)
            cfg.methods = ["rb_ccp"]
            cfg.physics.enabled = True
            _apply_override(cfg, key, value)
            factors = compute_physics_factors(cfg.vehicle, cfg.soil, cfg.physics)
            print(
                f"  {key}={value}: load={factors.wheel_load_n:.0f}N, "
                f"p={factors.contact_pressure_kpa:.1f}kPa, "
                f"moisture_factor={factors.moisture_factor:.2f}, "
                f"physics={factors.combined_factor:.2f}"
            )

            for wkt in wkt_files:
                try:
                    result = plan_field(wkt, cfg, project_root=project_root)
                    sel = result.selections[0]
                    m = metrics_from_selection(
                        result.field_name,
                        sel,
                        len(result.full_assessments),
                        len(result.informed_assessments),
                        cfg.selection.delta,
                        certificate=result.certificate,
                        runtime_full_assess_s=result.runtime_full_assess_s,
                        runtime_nr_pool_s=result.runtime_nr_pool_s,
                    )
                    row = m.to_dict()
                    row["sweep"] = sweep_name
                    row["sweep_param"] = key
                    row["sweep_value"] = value
                    rows.append(row)
                except Exception as exc:
                    print(f"    skip {Path(wkt).stem}: {exc}")

    out = project_root / config.output.results_dir / "physics_sensitivity_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n  saved {out} ({len(rows)} rows)")

    from src.visualization import plot_physics_sensitivity

    fig = project_root / config.output.figures_dir / "physics_sensitivity.png"
    plot_physics_sensitivity(rows, fig, dpi=config.output.dpi)
    print(f"  saved {fig}")
    _print_summary(rows)
    return 0


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    print("\nPhysics sensitivity summary (RB-CCP):")
    for sweep in sorted(set(r["sweep"] for r in rows)):
        values = sorted(set(float(r["sweep_value"]) for r in rows if r["sweep"] == sweep))
        print(f"  {sweep}:")
        for value in values:
            subset = [
                r for r in rows if r["sweep"] == sweep and float(r["sweep_value"]) == value
            ]
            n = len(subset)
            fb = sum(1 for r in subset if str(r["fallback"]).lower() in ("true", "1")) / n
            mr = sum(float(r["mean_risk"]) for r in subset) / n
            pf = sum(float(r["physics_factor"]) for r in subset) / n
            print(
                f"    value={value:g}: fallback={fb:.1%}, "
                f"mean_risk={mr:.3f}, physics_factor={pf:.2f}"
            )


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "nr_ccp_full.yaml")
    raise SystemExit(run_physics_sensitivity(cfg, ROOT))
