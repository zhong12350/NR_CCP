#!/usr/bin/env python3
"""Ablation study for risk-bound and risk-component contributions."""

from __future__ import annotations

import csv
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import AppConfig, load_config
from src.metrics import metrics_from_selection
from src.planner import plan_field


ABLATIONS = [
    # Main risk-bound baselines.
    ("baseline_rb", {"methods": ["rb_ccp"]}),
    ("no_risk_bound", {"methods": ["rb_ccp"], "selection.delta": 999.0}),
    ("weighted_only", {"methods": ["weighted"]}),
    ("no_informed_sampling", {"methods": ["nr_ccp"], "informed_sampling.enabled": False}),
    # Risk proxy component ablations. Each one keeps the RB-CCP selector fixed
    # so that changes are attributable to the risk model, not method mixing.
    ("no_headland", {"methods": ["rb_ccp"], "risk_field.headland_base": "inner_base"}),
    ("no_hotspot", {"methods": ["rb_ccp"], "risk_field.auto_hotspots": False, "risk_field.gaussians": []}),
    ("no_pass_count", {"methods": ["rb_ccp"], "risk_field.use_pass_count": False}),
    ("no_repeat", {"methods": ["rb_ccp"], "risk_field.repeat_penalty": 0.0}),
    ("no_turn", {"methods": ["rb_ccp"], "risk_field.turning_penalty": 0.0}),
]


RISK_COMPONENT = {
    "baseline_rb": "all",
    "no_headland": "headland",
    "no_hotspot": "hotspot",
    "no_pass_count": "pass_count",
    "no_repeat": "repeat",
    "no_turn": "turn",
}


def _apply_override(cfg: AppConfig, key: str, value) -> None:
    parts = key.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if value == "inner_base":
        value = cfg.risk_field.inner_base
    setattr(obj, parts[-1], value)


def run_ablation(config: AppConfig, project_root: Path) -> int:
    pattern = str(project_root / config.batch.field_glob)
    wkt_files = sorted(glob(pattern))[: min(config.batch.max_fields, 30)]
    rows: list[dict] = []

    for name, overrides in ABLATIONS:
        print(f"\n=== Ablation: {name} ===")
        cfg = deepcopy(config)
        for k, v in overrides.items():
            _apply_override(cfg, k, v)

        for wkt in wkt_files:
            try:
                result = plan_field(wkt, cfg, project_root=project_root)
                for sel in result.selections:
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
                    row["ablation"] = name
                    row["risk_component_removed"] = RISK_COMPONENT.get(name, "")
                    rows.append(row)
            except Exception as exc:
                print(f"  skip {Path(wkt).stem}: {exc}")

    out = project_root / config.output.results_dir / "ablation_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n  saved {out}")
    return 0


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "nr_ccp_full.yaml")
    raise SystemExit(run_ablation(cfg, ROOT))
