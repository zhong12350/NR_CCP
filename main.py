"""NR-CCP full application entry."""

from __future__ import annotations

import csv
import sys
import time
from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np

from src.config_loader import AppConfig, load_config
from src.metrics import MethodMetrics, metrics_from_selection
from src.planner import FieldPlanResult, plan_field
from src.visualization import plot_field_and_path, plot_method_comparison


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_single_field(
    config: AppConfig,
    project_root: Path,
    wkt_path: Path | None = None,
    output_subdir: str | None = None,
    save_figures: bool = True,
    verbose: bool = True,
) -> tuple[FieldPlanResult, list[MethodMetrics]]:
    wkt = wkt_path or (project_root / config.field.wkt_path)
    t0 = time.perf_counter()
    result = plan_field(wkt, config, project_root=project_root)
    elapsed = time.perf_counter() - t0

    metrics_list = [
        metrics_from_selection(
            result.field_name,
            sel,
            len(result.full_assessments),
            len(result.informed_assessments),
            config.selection.delta,
            certificate=result.certificate,
            runtime_full_assess_s=result.runtime_full_assess_s,
            runtime_nr_pool_s=result.runtime_nr_pool_s,
        )
        for sel in result.selections
    ]

    if save_figures:
        fig_base = project_root / config.output.figures_dir
        res_base = project_root / config.output.results_dir
        if output_subdir:
            fig_base = fig_base / output_subdir
            res_base = res_base / output_subdir

        for sel, m in zip(result.selections, metrics_list):
            fig_path = fig_base / f"{result.field_name}_{sel.method}.png"
            plot_field_and_path(
                result.grid, result.risk, sel, m, fig_path, dpi=config.output.dpi
            )
            print(f"  saved {fig_path.relative_to(project_root)}")

        comparison_path = fig_base / f"{result.field_name}_comparison.png"
        plot_method_comparison(metrics_list, comparison_path, dpi=config.output.dpi)
        print(f"  saved {comparison_path.relative_to(project_root)}")

        csv_path = res_base / f"{result.field_name}_metrics.csv"
        _write_csv([m.to_dict() for m in metrics_list], csv_path)
        print(f"  saved {csv_path.relative_to(project_root)}")

    if verbose:
        _print_summary(result.field_name, metrics_list, result, elapsed)
    return result, metrics_list


def run_batch(config: AppConfig, project_root: Path) -> list[dict]:
    pattern = str(project_root / config.batch.field_glob)
    wkt_files = sorted(glob(pattern))[: config.batch.max_fields]
    if not wkt_files:
        raise FileNotFoundError(f"No WKT fields matched: {pattern}")

    all_rows: list[dict] = []
    print(f"\n=== Batch: {len(wkt_files)} fields | figures={config.batch.save_figures} ===")
    t0 = time.perf_counter()

    for i, wkt in enumerate(wkt_files, start=1):
        if i == 1 or i % config.batch.progress_every == 0 or i == len(wkt_files):
            print(f"[{i}/{len(wkt_files)}] {Path(wkt).stem}")
        try:
            _, metrics = run_single_field(
                config,
                project_root,
                wkt_path=Path(wkt),
                output_subdir="batch" if config.batch.save_figures else None,
                save_figures=config.batch.save_figures,
                verbose=False,
            )
            all_rows.extend(m.to_dict() for m in metrics)
        except Exception as exc:
            print(f"  SKIP {Path(wkt).stem}: {exc}")

    elapsed = time.perf_counter() - t0
    results_dir = project_root / config.output.results_dir
    batch_csv = results_dir / "batch_results.csv"
    _write_csv(all_rows, batch_csv)
    print(f"\n  saved {batch_csv.relative_to(project_root)} ({elapsed:.1f}s)")

    _write_method_summary(all_rows, results_dir / "method_summary.csv", config.selection.delta)
    _write_violation_summary(all_rows, results_dir / "violation_summary.csv")
    _write_certificate_summary(all_rows, results_dir / "feasibility_certificates.csv")
    return all_rows


def run_advisor_demo(config: AppConfig, project_root: Path, n_fields: int = 5) -> None:
    pattern = str(project_root / config.batch.field_glob)
    wkt_files = sorted(glob(pattern))[:n_fields]
    demo_dir = project_root / config.output.advisor_demo_dir
    demo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Advisor demo: {len(wkt_files)} fields → {demo_dir} ===")
    for wkt in wkt_files:
        result = plan_field(wkt, config, project_root=project_root)
        for sel in result.selections:
            m = metrics_from_selection(
                result.field_name,
                sel,
                len(result.full_assessments),
                len(result.informed_assessments),
                config.selection.delta,
                certificate=result.certificate,
                runtime_full_assess_s=result.runtime_full_assess_s,
                runtime_nr_pool_s=result.runtime_nr_pool_s,
            )
            out = demo_dir / f"{result.field_name}_{sel.method}.png"
            plot_field_and_path(
                result.grid, result.risk, sel, m, out, dpi=config.output.dpi
            )
            print(f"  saved {out.relative_to(project_root)}")


def _write_method_summary(rows: list[dict], path: Path, delta: float) -> None:
    methods = sorted(set(r["method"] for r in rows))
    summary = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        n = len(subset)
        summary.append(
            {
                "method": method,
                "delta": delta,
                "n_fields": n,
                "mean_path_length_m": sum(r["path_length_m"] for r in subset) / n,
                "mean_compaction_cost": sum(r["compaction_cost"] for r in subset) / n,
                "mean_risk": sum(r["mean_risk"] for r in subset) / n,
                "fallback_rate": sum(1 for r in subset if r["fallback"]) / n,
            }
        )
    _write_csv(summary, path)
    print(f"  saved {path}")


def _write_certificate_summary(rows: list[dict], path: Path) -> None:
    seen = set()
    cert_rows = []
    for r in rows:
        if r["method"] != "rb_ccp" or r["field_name"] in seen:
            continue
        seen.add(r["field_name"])
        cert_rows.append(
            {
                "field_name": r["field_name"],
                "delta": r["delta"],
                "field_area_m2": r.get("field_area_m2", 0),
                "aspect_ratio": r.get("aspect_ratio", 0),
                "boundary_complexity": r.get("boundary_complexity", 0),
                "cert_best_coverage": r.get("cert_best_coverage", 0),
                "cert_min_violation": r.get("cert_min_violation", 0),
                "num_candidates": r.get("num_candidates", 0),
                "num_informed_candidates": r.get("num_informed_candidates", 0),
            }
        )
    _write_csv(cert_rows, path)
    print(f"  saved {path}")


def _write_violation_summary(rows: list[dict], path: Path) -> None:
    methods = sorted(set(r["method"] for r in rows))
    summary = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        n = len(subset)
        summary.append(
            {
                "method": method,
                "n_fields": n,
                "fallback_count": sum(1 for r in subset if r["fallback"]),
                "fallback_rate": sum(1 for r in subset if r["fallback"]) / n,
                "mean_violation": sum(r["violation"] for r in subset) / n,
            }
        )
    _write_csv(summary, path)
    print(f"  saved {path}")


def _print_summary(
    field_name: str,
    metrics: list[MethodMetrics],
    result: FieldPlanResult,
    elapsed: float,
) -> None:
    print(
        f"\n--- {field_name} | full={len(result.full_assessments)} "
        f"informed={len(result.informed_assessments)} | {elapsed:.1f}s ---"
    )
    for m in metrics:
        fb = " [FALLBACK]" if m.fallback else ""
        print(
            f"  {m.method}{fb}: θ={m.angle_deg:.0f}°, "
            f"L={m.path_length_m:.0f}m, C={m.compaction_cost:.0f}, "
            f"mean_risk={m.mean_risk:.3f}, cov={m.coverage_rate:.1%}"
        )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = _project_root()
    config_path = root / "configs" / "nr_ccp_full.yaml"
    mode = "single"

    if argv:
        if argv[0] in (
            "batch",
            "advisor",
            "analyze",
            "delta_sweep",
            "ablation",
            "train_sampler",
            "paper_tables",
            "import_f2c",
            "budget_experiment",
        ):
            mode = argv[0]
            argv = argv[1:]
        elif argv[0].endswith(".yaml"):
            config_path = Path(argv[0])
            if not config_path.is_absolute():
                config_path = root / config_path
            argv = argv[1:]

    print(f"NR-CCP | mode={mode} | config: {config_path}")
    config = load_config(config_path)
    np.random.seed(config.seed)

    if mode == "batch":
        run_batch(config, root)
    elif mode == "advisor":
        run_advisor_demo(config, root)
    elif mode == "analyze":
        import importlib.util

        script = root / "scripts" / "analyze_results.py"
        spec = importlib.util.spec_from_file_location("analyze_results", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main(root, config_path)
    elif mode == "delta_sweep":
        from scripts.run_delta_sweep import run_delta_sweep

        return run_delta_sweep(config, root)
    elif mode == "ablation":
        from scripts.run_ablation import run_ablation

        return run_ablation(config, root)
    elif mode == "train_sampler":
        from scripts.train_informed_sampler import train_sampler

        return train_sampler(config, root)
    elif mode == "paper_tables":
        from scripts.generate_paper_tables import generate_paper_tables

        return generate_paper_tables(root, config)
    elif mode == "import_f2c":
        from scripts.import_fields2cover import import_fields2cover

        return import_fields2cover(config, root)
    elif mode == "budget_experiment":
        from scripts.run_budget_experiment import run_budget_experiment

        return run_budget_experiment(config, root)
    else:
        wkt = Path(argv[0]) if argv else None
        run_single_field(config, root, wkt_path=wkt)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
