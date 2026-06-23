"""Matplotlib visualization for RB-CCP V1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon

from src.fields import RiskField
from src.geometry import FieldGrid
from src.metrics import MethodMetrics
from src.selectors import SelectionResult


METHOD_LABELS = {
    "naive": "Naive min-length",
    "weighted": "Soil2Cover-like weighted",
    "rb_ccp": "RB-CCP (ours)",
    "nr_ccp": "NR-CCP (informed)",
    "fields2cover": "Fields2Cover-like",
}


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _draw_polygon(ax, poly: Polygon, **kwargs) -> None:
    x, y = poly.exterior.xy
    ax.add_patch(MplPolygon(np.column_stack([x, y]), closed=True, **kwargs))


def plot_field_and_path(
    grid: FieldGrid,
    risk: RiskField,
    selection: SelectionResult,
    metrics: MethodMetrics,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Overlay risk heatmap, headland, and selected coverage path."""
    fig, ax = plt.subplots(figsize=(10, 7))
    minx, miny, maxx, maxy = grid.geometry.bounds
    extent = [minx, maxx, miny, maxy]

    base = np.ma.masked_where(~grid.outer_mask, risk.values)
    im = ax.imshow(
        base,
        origin="lower",
        extent=extent,
        cmap="YlOrRd",
        alpha=0.85,
        vmin=0,
        vmax=1,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Planning risk R(x)")

    _draw_polygon(
        ax,
        grid.geometry.inner,
        fill=False,
        edgecolor="white",
        linewidth=1.2,
        linestyle="--",
        label="Inner workable",
    )
    _draw_polygon(
        ax,
        grid.geometry.outer,
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        label="Field boundary",
    )

    waypoints = selection.assessment.candidate.waypoints
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    ax.plot(xs, ys, color="royalblue", linewidth=1.5, label="Coverage path")
    ax.scatter(xs[0], ys[0], c="lime", s=60, zorder=5, label="Start")
    ax.scatter(xs[-1], ys[-1], c="black", s=40, zorder=5, label="End")

    fb = " [FALLBACK]" if selection.fallback else ""
    title = (
        f"{grid.geometry.name} | {_label(selection.method)}{fb}\n"
        f"θ={metrics.angle_deg:.0f}°, L={metrics.path_length_m:.0f}m, "
        f"mean_risk={metrics.mean_risk:.3f} (Δ={metrics.delta}), "
        f"C={metrics.compaction_cost:.0f}, cov={metrics.coverage_rate:.1%}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(
    metrics_list: list[MethodMetrics],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Bar chart comparing methods on one field."""
    labels = [_label(m.method) for m in metrics_list]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].bar(x, [m.path_length_m for m in metrics_list], color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Path length (m)")
    axes[0].set_title("L(π)")

    axes[1].bar(x, [m.compaction_cost for m in metrics_list], color="indianred")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Compaction cost C")
    axes[1].set_title("C(π)")

    axes[2].bar(x, [m.mean_risk for m in metrics_list], color="seagreen")
    axes[2].axhline(
        metrics_list[0].delta if metrics_list else 0.38,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Δ bound",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].set_ylabel("Mean risk")
    axes[2].set_title("mean_risk(π)")
    axes[2].legend(fontsize=8)

    field_name = metrics_list[0].field_name if metrics_list else "field"
    fig.suptitle(f"RB-CCP V1 Comparison — {field_name}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(
    rows: list[dict],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Pareto scatter: path length vs compaction cost."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "naive": "steelblue",
        "weighted": "darkorange",
        "rb_ccp": "seagreen",
        "nr_ccp": "purple",
        "fields2cover": "gray",
    }
    markers = {
        "naive": "o",
        "weighted": "s",
        "rb_ccp": "^",
        "nr_ccp": "D",
        "fields2cover": "x",
    }

    for method in colors:
        pts = [r for r in rows if r["method"] == method]
        if not pts:
            continue
        ax.scatter(
            [p["path_length_m"] for p in pts],
            [p["compaction_cost"] for p in pts],
            c=colors[method],
            marker=markers[method],
            s=50,
            alpha=0.75,
            label=_label(method),
        )

    ax.set_xlabel("Path length L (m)")
    ax.set_ylabel("Compaction cost C")
    ax.set_title("Pareto: Length vs Compaction Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_batch_summary(
    summary_rows: list[dict],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Grouped bar chart of batch method summary."""
    methods = [r["method"] for r in summary_rows]
    labels = [_label(m) for m in methods]
    x = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x, [r["mean_path_length_m"] for r in summary_rows], color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Mean path length (m)")

    axes[1].bar(x, [r["mean_risk"] for r in summary_rows], color="seagreen")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Mean risk")
    axes[1].set_title(f"Δ = {summary_rows[0].get('delta', 0.38)}")

    fig.suptitle("Batch Method Summary", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_delta_violation(
    violation_rows: list[dict],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Fallback rate by method."""
    methods = [r["method"] for r in violation_rows]
    labels = [_label(m) for m in methods]
    rates = [r["fallback_rate"] for r in violation_rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, rates, color="indianred")
    ax.set_ylabel("Fallback rate")
    ax.set_title("Δ constraint violation / fallback rate")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_delta_sweep(rows: list[dict], out_path: Path, dpi: int = 150) -> None:
    """Δ vs fallback rate, path length, and mean risk."""
    if not rows:
        return
    deltas = sorted(set(float(r["delta"]) for r in rows))
    methods = sorted(set(r["method"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for method in methods:
        fb_rates = []
        lengths = []
        risks = []
        for d in deltas:
            subset = [r for r in rows if r["method"] == method and float(r["delta"]) == d]
            if not subset:
                fb_rates.append(0.0)
                lengths.append(0.0)
                risks.append(0.0)
                continue
            fb = [r["fallback"] if isinstance(r["fallback"], bool) else str(r["fallback"]).lower() in ("true", "1") for r in subset]
            fb_rates.append(sum(fb) / len(subset))
            lengths.append(sum(float(r["path_length_m"]) for r in subset) / len(subset))
            risks.append(sum(float(r["mean_risk"]) for r in subset) / len(subset))
        axes[0].plot(deltas, fb_rates, marker="o", label=_label(method))
        axes[1].plot(deltas, lengths, marker="o", label=_label(method))
        axes[2].plot(deltas, risks, marker="o", label=_label(method))

    axes[0].set_xlabel("Δ")
    axes[0].set_ylabel("Fallback rate")
    axes[0].set_title("Δ vs fallback")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Δ")
    axes[1].set_ylabel("Mean path length (m)")
    axes[1].set_title("Δ vs path length")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Δ")
    axes[2].set_ylabel("Mean risk")
    axes[2].set_title("Δ vs mean risk")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("NR-CCP Δ Sensitivity", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_fallback_grouping(
    grouping_rows: list[dict],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Fallback rate by country / area / aspect ratio bins."""
    if not grouping_rows:
        return
    country_rows = [r for r in grouping_rows if r.get("group_type") == "country"]
    if not country_rows:
        return

    prefixes = sorted(set(r["group_value"] for r in country_rows))
    methods = sorted(set(r["method"] for r in country_rows))
    x = np.arange(len(prefixes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, method in enumerate(methods):
        rates = []
        for p in prefixes:
            subset = [r for r in country_rows if r["group_value"] == p and r["method"] == method]
            rates.append(float(subset[0]["fallback_rate"]) if subset else 0.0)
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=_label(method))

    ax.set_xticks(x)
    ax.set_xticklabels(prefixes)
    ax.set_ylabel("Fallback rate")
    ax.set_xlabel("Country prefix")
    ax.set_title("Fallback by field origin (ee / lt / nl)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_comparison(
    runtime_rows: list[dict],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """NR pool size fraction vs full enumeration."""
    if not runtime_rows:
        return
    fracs = [float(r["nr_pool_fraction"]) for r in runtime_rows]
    full_t = [float(r["runtime_full_assess_s"]) for r in runtime_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(fracs, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("NR pool / full pool fraction")
    axes[0].set_ylabel("Field count")
    axes[0].set_title("Informed pool compression")

    axes[1].scatter(fracs, full_t, alpha=0.5, s=20, c="indianred")
    axes[1].set_xlabel("NR pool fraction")
    axes[1].set_ylabel("Full assess runtime (s)")
    axes[1].set_title("Runtime vs pool size")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_risk_decomposition(
    risk_rows: list[dict],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Stacked bar of decomposed risk components."""
    if not risk_rows:
        return
    methods = [r["method"] for r in risk_rows]
    labels = [_label(m) for m in methods]
    components = ["headland", "hotspot", "pass_count", "repeat", "turn"]
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#9b59b6"]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(methods))
    for comp, color in zip(components, colors):
        vals = [float(r[f"mean_{comp}_cost"]) for r in risk_rows]
        ax.bar(x, vals, bottom=bottom, label=comp.replace("_", " "), color=color)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean component cost")
    ax.set_title("Risk proxy decomposition")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_summary(
    ablation_rows: list[dict],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Ablation fallback rate and mean risk."""
    if not ablation_rows:
        return
    ablations = sorted(set(r["ablation"] for r in ablation_rows))
    fb_rates = []
    risks = []
    for name in ablations:
        subset = [r for r in ablation_rows if r["ablation"] == name]
        n = len(subset)
        fb = sum(1 for r in subset if str(r.get("fallback", "")).lower() in ("true", "1"))
        fb_rates.append(fb / n if n else 0.0)
        risks.append(sum(float(r["mean_risk"]) for r in subset) / n if n else 0.0)

    x = np.arange(len(ablations))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x, fb_rates, color="indianred")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ablations, rotation=20, ha="right")
    axes[0].set_ylabel("Fallback rate")
    axes[0].set_title("Ablation: fallback")

    axes[1].bar(x, risks, color="seagreen")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ablations, rotation=20, ha="right")
    axes[1].set_ylabel("Mean risk")
    axes[1].set_title("Ablation: mean risk")
    fig.suptitle("NR-CCP Ablation Study", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_budget_experiment(rows: list[dict], out_path: Path, dpi: int = 150) -> None:
    """Fixed-budget NR-prior vs uniform/random comparison."""
    if not rows:
        return

    def as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")

    strategies = [s for s in ("nr_prior", "uniform", "random") if any(r["strategy"] == s for r in rows)]
    budgets = sorted(
        {
            int(r["budget"])
            for r in rows
            if r.get("strategy") in strategies
        }
    )
    if not strategies or not budgets:
        return

    colors = {"nr_prior": "purple", "uniform": "steelblue", "random": "gray"}
    labels = {"nr_prior": "NR prior", "uniform": "Uniform-k", "random": "Random-k"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for strategy in strategies:
        fb_rates = []
        match_rates = []
        risk_gaps = []
        for budget in budgets:
            subset = [
                r
                for r in rows
                if r["strategy"] == strategy and int(r["budget"]) == budget
            ]
            if not subset:
                fb_rates.append(0.0)
                match_rates.append(0.0)
                risk_gaps.append(0.0)
                continue
            fb_rates.append(sum(1 for r in subset if as_bool(r["fallback"])) / len(subset))
            match_rates.append(sum(1 for r in subset if as_bool(r["same_as_oracle"])) / len(subset))
            risk_gaps.append(
                sum(float(r["risk_gap_vs_oracle"]) for r in subset) / len(subset)
            )

        axes[0].plot(budgets, fb_rates, marker="o", label=labels[strategy], color=colors[strategy])
        axes[1].plot(budgets, match_rates, marker="o", label=labels[strategy], color=colors[strategy])
        axes[2].plot(budgets, risk_gaps, marker="o", label=labels[strategy], color=colors[strategy])

    axes[0].set_title("Fallback Rate")
    axes[0].set_ylabel("Rate")
    axes[1].set_title("Oracle Angle Match")
    axes[1].set_ylabel("Rate")
    axes[2].set_title("Mean Risk Gap vs RB-full")
    axes[2].set_ylabel("Risk gap")
    for ax in axes:
        ax.set_xlabel("Candidate budget k")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Fixed-Budget Candidate Search", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
