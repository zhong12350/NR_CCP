#!/usr/bin/env python3
"""Fixed-budget experiment for the NR-CCP informed search contribution."""

from __future__ import annotations

import csv
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.assessor import PathAssessment, assess_all_candidates, build_feasibility_certificate
from src.candidates import CandidatePath, enumerate_candidates
from src.config_loader import AppConfig, load_config
from src.fields import build_risk_field
from src.geometry import load_field_from_wkt
from src.metrics import metrics_from_selection
from src.physics import compute_physics_factors
from src.risk_search import rank_candidates_by_prior
from src.selectors import SelectionResult, select_rb_ccp
from src.visualization import plot_budget_experiment


def _parse_budgets() -> list[int]:
    raw = os.getenv("NR_CCP_BUDGETS", "4,6,8,12")
    budgets = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            budgets.append(int(part))
    return sorted(set(budgets))


def _max_fields(config: AppConfig) -> int:
    raw = os.getenv("NR_CCP_BUDGET_FIELDS")
    if raw:
        return min(config.batch.max_fields, int(raw))
    return min(config.batch.max_fields, 50)


def _uniform_subset(candidates: list[CandidatePath], k: int) -> list[CandidatePath]:
    if k >= len(candidates):
        return candidates
    ordered = sorted(candidates, key=lambda c: c.angle_deg)
    idxs = np.linspace(0, len(ordered) - 1, k, dtype=int)
    return [ordered[int(i)] for i in idxs]


def _random_subset(candidates: list[CandidatePath], k: int, seed: int) -> list[CandidatePath]:
    if k >= len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(candidates), size=k, replace=False).tolist())
    return [candidates[i] for i in idxs]


def _assess_subset(
    subset: list[CandidatePath],
    all_assessments: list[PathAssessment],
) -> list[PathAssessment]:
    by_angle = {round(a.angle_deg % 180.0, 2): a for a in all_assessments}
    return [
        by_angle[round(c.angle_deg % 180.0, 2)]
        for c in subset
        if round(c.angle_deg % 180.0, 2) in by_angle
    ]


def _row_from_selection(
    field_name: str,
    strategy: str,
    budget: int,
    selection: SelectionResult,
    oracle: SelectionResult,
    num_full: int,
    num_eval: int,
    delta: float,
    certificate,
) -> dict:
    m = metrics_from_selection(
        field_name,
        SelectionResult(
            strategy,
            selection.assessment,
            selection.fallback,
            selection.violation,
            selection.candidate_pool,
        ),
        num_full,
        num_eval,
        delta,
        certificate=certificate,
    ).to_dict()
    m["strategy"] = strategy
    m["budget"] = budget
    m["num_evaluated"] = num_eval
    m["pool_fraction"] = num_eval / max(num_full, 1)
    m["oracle_angle_deg"] = oracle.assessment.angle_deg
    m["oracle_fallback"] = oracle.fallback
    m["oracle_violation"] = oracle.violation
    m["oracle_mean_risk"] = oracle.assessment.mean_risk
    m["oracle_path_length_m"] = oracle.assessment.path_length_m
    m["risk_gap_vs_oracle"] = selection.assessment.mean_risk - oracle.assessment.mean_risk
    m["path_gap_vs_oracle"] = selection.assessment.path_length_m - oracle.assessment.path_length_m
    m["same_as_oracle"] = abs(selection.assessment.angle_deg - oracle.assessment.angle_deg) < 1e-9
    return m


def run_budget_experiment(config: AppConfig, project_root: Path) -> int:
    budgets = _parse_budgets()
    pattern = str(project_root / config.batch.field_glob)
    wkt_files = sorted(glob(pattern))[: _max_fields(config)]
    rows: list[dict] = []

    print(
        f"Fixed-budget NR experiment on {len(wkt_files)} fields, "
        f"budgets={budgets}"
    )

    for i, wkt in enumerate(wkt_files, start=1):
        if i == 1 or i % config.batch.progress_every == 0 or i == len(wkt_files):
            print(f"  [{i}/{len(wkt_files)}] {Path(wkt).stem}")

        try:
            grid = load_field_from_wkt(
                wkt,
                config.headland.width_m,
                cell_size_m=config.field.cell_size_m,
                auto_cell_size=config.field.auto_cell_size,
            )
            risk = build_risk_field(grid, config.risk_field, config.planner)
            full_candidates = enumerate_candidates(grid, config.planner)
            physics_factors = compute_physics_factors(
                config.vehicle, config.soil, config.physics
            )
            full_assessments = assess_all_candidates(
                full_candidates,
                grid,
                risk,
                config.risk_field,
                config.planner,
                config.selection.lambda_weighted,
                config.selection.beta_rb_ccp,
                config.planner.min_coverage,
                physics_factors=physics_factors,
            )
            if not full_assessments:
                continue

            field_name = Path(wkt).stem
            oracle = select_rb_ccp(
                full_assessments,
                config.selection.delta,
                config.selection.beta_rb_ccp,
            )
            certificate = build_feasibility_certificate(
                field_name,
                grid,
                full_assessments,
                [],
                config.selection.delta,
            )
            full_by_angle = {
                round(a.angle_deg % 180.0, 2): a for a in full_assessments
            }
            valid_candidates = [
                c
                for c in full_candidates
                if round(c.angle_deg % 180.0, 2) in full_by_angle
            ]
            prior_ranked = rank_candidates_by_prior(
                valid_candidates,
                grid,
                risk,
                config.planner,
                config.informed_sampling,
                seed=config.seed,
            )

            rows.append(
                _row_from_selection(
                    field_name,
                    "rb_full",
                    len(valid_candidates),
                    oracle,
                    oracle,
                    len(valid_candidates),
                    len(valid_candidates),
                    config.selection.delta,
                    certificate,
                )
            )

            for budget in budgets:
                subsets = {
                    "nr_prior": prior_ranked[:budget],
                    "uniform": _uniform_subset(valid_candidates, budget),
                    "random": _random_subset(
                        valid_candidates, budget, config.seed + i * 1000 + budget
                    ),
                }
                for strategy, subset in subsets.items():
                    assessed = _assess_subset(subset, full_assessments)
                    if not assessed:
                        continue
                    sel = select_rb_ccp(
                        assessed,
                        config.selection.delta,
                        config.selection.beta_nr_ccp,
                    )
                    sel = SelectionResult(strategy, sel.assessment, sel.fallback, sel.violation, strategy)
                    rows.append(
                        _row_from_selection(
                            field_name,
                            strategy,
                            budget,
                            sel,
                            oracle,
                            len(valid_candidates),
                            len(assessed),
                            config.selection.delta,
                            certificate,
                        )
                    )
        except Exception as exc:
            print(f"  skip {Path(wkt).stem}: {exc}")

    out = project_root / config.output.results_dir / "budget_experiment_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n  saved {out} ({len(rows)} rows)")
    fig = project_root / config.output.figures_dir / "budget_experiment.png"
    plot_budget_experiment(rows, fig, dpi=config.output.dpi)
    print(f"  saved {fig}")
    _print_summary(rows)
    return 0


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    print("\nBudget experiment summary:")
    keys = sorted({(r["strategy"], int(r["budget"])) for r in rows})
    for strategy, budget in keys:
        subset = [r for r in rows if r["strategy"] == strategy and int(r["budget"]) == budget]
        n = len(subset)
        fallback_rate = sum(1 for r in subset if r["fallback"]) / n
        same_rate = sum(1 for r in subset if r["same_as_oracle"]) / n
        risk_gap = sum(float(r["risk_gap_vs_oracle"]) for r in subset) / n
        path_gap = sum(float(r["path_gap_vs_oracle"]) for r in subset) / n
        print(
            f"  {strategy:8s} k={budget:2d}: "
            f"fallback={fallback_rate:.1%}, oracle_match={same_rate:.1%}, "
            f"risk_gap={risk_gap:+.3f}, path_gap={path_gap:+.0f}m"
        )


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "nr_ccp_full.yaml")
    raise SystemExit(run_budget_experiment(cfg, ROOT))
