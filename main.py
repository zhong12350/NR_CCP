"""Application pipeline: load config → plan → evaluate → visualize."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from src.config_loader import AppConfig, load_config
from src.fields import build_risk_field
from src.geometry import build_field_grid
from src.metrics import PathMetrics, evaluate_all
from src.planner import PlanResult, plan_all
from src.visualization import plot_field_and_path, plot_metrics_comparison


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def run_pipeline(config: AppConfig, project_root: Path | None = None) -> tuple[list[PlanResult], list[PathMetrics]]:
    root = project_root or _project_root()
    np.random.seed(config.seed)

    grid = build_field_grid(config.field)
    risk = build_risk_field(grid, config.risk_field)
    tool_radius = config.planner.strip_width_m / 2.0

    results = plan_all(config.methods, grid, risk, config.planner)
    metrics = evaluate_all(results, grid, risk, tool_radius)

    figures_dir = root / config.output.figures_dir
    results_dir = root / config.output.results_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for result, m in zip(results, metrics):
        fig_path = figures_dir / f"{result.method}.png"
        plot_field_and_path(grid, risk, result, m, fig_path, dpi=config.output.dpi)
        print(f"  saved {fig_path.relative_to(root)}")

    comparison_path = figures_dir / "comparison.png"
    plot_metrics_comparison(metrics, comparison_path, dpi=config.output.dpi)
    print(f"  saved {comparison_path.relative_to(root)}")

    csv_path = results_dir / "metrics.csv"
    _write_metrics_csv(metrics, csv_path)
    print(f"  saved {csv_path.relative_to(root)}")

    _print_summary(metrics)
    return results, metrics


def _write_metrics_csv(metrics: list[PathMetrics], path: Path) -> None:
    if not metrics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [m.to_dict() for m in metrics]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(metrics: list[PathMetrics]) -> None:
    print("\n--- Metrics summary ---")
    for m in metrics:
        print(
            f"  {m.method}: "
            f"length={m.path_length_m:.1f}m, "
            f"risk×length={m.risk_length_cost:.1f}, "
            f"mean_risk={m.mean_risk:.3f}, "
            f"coverage={m.coverage_rate:.1%}"
        )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = _project_root()
    config_path = root / "configs" / "default.yaml"
    if argv:
        config_path = Path(argv[0])
        if not config_path.is_absolute():
            config_path = root / config_path

    print(f"NR-CCP Demo | config: {config_path}")
    config = load_config(config_path)
    run_pipeline(config, root)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
