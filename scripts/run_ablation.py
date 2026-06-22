#!/usr/bin/env python3
"""Five ablation groups from proposal §11.2."""

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
    ("baseline", {}),
    ("no_risk_bound", {"selection.delta": 999.0}),
    ("weighted_only", {"methods": ["weighted"]}),
    ("no_pass_count", {"risk_field.use_pass_count": False}),
    ("no_informed_sampling", {"informed_sampling.enabled": False, "methods": ["nr_ccp"]}),
]


def _apply_override(cfg: AppConfig, key: str, value) -> None:
    parts = key.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
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
        if name == "weighted_only":
            cfg.methods = ["weighted"]
        elif name == "no_informed_sampling":
            cfg.methods = ["nr_ccp"]

        for wkt in wkt_files:
            try:
                result = plan_field(wkt, cfg)
                for sel in result.selections:
                    m = metrics_from_selection(
                        result.field_name,
                        sel,
                        len(result.full_assessments),
                        len(result.informed_assessments),
                        cfg.selection.delta,
                    )
                    row = m.to_dict()
                    row["ablation"] = name
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
