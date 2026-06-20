"""Matplotlib visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fields import RiskField
from src.geometry import FieldGrid
from src.metrics import PathMetrics
from src.planner import PlanResult


METHOD_LABELS = {
    "uniform_boustrophedon": "Uniform Boustrophedon",
    "risk_aware_reordering": "Risk-Aware Reordering",
    "informed_strip_selection": "Informed Strip Selection",
}


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def plot_field_and_path(
    grid: FieldGrid,
    risk: RiskField,
    result: PlanResult,
    metrics: PathMetrics,
    out_path: Path,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [0, grid.width_m, 0, grid.height_m]
    im = ax.imshow(
        risk.values,
        origin="lower",
        extent=extent,
        cmap="YlOrRd",
        alpha=0.85,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Compaction risk")

    if grid.obstacle_mask.any():
        obs = np.ma.masked_where(~grid.obstacle_mask, np.ones_like(risk.values))
        ax.imshow(obs, origin="lower", extent=extent, cmap="gray", alpha=0.5, vmin=0, vmax=1)

    xs = [p[0] for p in result.waypoints]
    ys = [p[1] for p in result.waypoints]
    ax.plot(xs, ys, color="royalblue", linewidth=1.5, label="Coverage path")
    ax.scatter(xs[0], ys[0], c="lime", s=60, zorder=5, label="Start")
    ax.scatter(xs[-1], ys[-1], c="black", s=40, zorder=5, label="End")

    title = (
        f"{_label(result.method)}\n"
        f"L={metrics.path_length_m:.1f}m, "
        f"risk×length={metrics.risk_length_cost:.1f}, "
        f"coverage={metrics.coverage_rate:.1%}"
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, grid.width_m)
    ax.set_ylim(0, grid.height_m)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(
    metrics_list: list[PathMetrics],
    out_path: Path,
    dpi: int = 150,
) -> None:
    labels = [_label(m.method) for m in metrics_list]
    x = np.arange(len(labels))
    width = 0.35

    path_lengths = [m.path_length_m for m in metrics_list]
    risk_costs = [m.risk_length_cost for m in metrics_list]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(x, path_lengths, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Path length (m)")
    axes[0].set_title("Total path length")

    axes[1].bar(x, risk_costs, color="indianred")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Risk × length cost")
    axes[1].set_title("Risk-weighted traversal cost")

    fig.suptitle("NR-CCP Demo: Method Comparison", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
