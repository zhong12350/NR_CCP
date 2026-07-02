#!/usr/bin/env python3
"""Train the NR-CCP neural informed angle sampler with field-level validation."""

from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.assessor import assess_all_candidates
from src.candidates import enumerate_candidates
from src.config_loader import AppConfig, load_config
from src.fields import build_risk_field
from src.geometry import load_field_from_wkt
from src.informed_sampling import InformedAngleSampler, default_model, save_model
from src.physics import compute_physics_factors


def _collect_dataset(config: AppConfig, project_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return feature matrix, targets, and integer field ids."""
    pattern = str(project_root / config.batch.field_glob)
    wkt_files = sorted(glob(pattern))[: min(config.batch.max_fields, 80)]
    sampler = InformedAngleSampler(config.informed_sampling, config.seed)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    field_ids: list[int] = []

    print(f"Collecting neural sampler dataset from {len(wkt_files)} fields...")
    for field_id, wkt in enumerate(wkt_files):
        try:
            grid = load_field_from_wkt(
                wkt,
                config.headland.width_m,
                cell_size_m=config.field.cell_size_m,
                auto_cell_size=config.field.auto_cell_size,
            )
            risk = build_risk_field(grid, config.risk_field, config.planner)
            full = enumerate_candidates(grid, config.planner)
            assessed = assess_all_candidates(
                full,
                grid,
                risk,
                config.risk_field,
                config.planner,
                config.selection.lambda_weighted,
                config.selection.beta_rb_ccp,
                config.planner.min_coverage,
                physics_factors=compute_physics_factors(
                    config.vehicle, config.soil, config.physics
                ),
            )
            if not assessed:
                continue

            max_len = max(a.path_length_m for a in assessed) or 1.0
            max_cost = max(a.compaction_cost for a in assessed) or 1.0
            for a in assessed:
                x = sampler.feature_vector(
                    grid, risk, a.angle_deg, config.planner.swath_width_m
                )
                violation = max(0.0, a.bound_risk - config.selection.delta)
                # Higher is better: feasible, low CVaR risk, low compaction, short path.
                target = -(
                    2.0 * violation
                    + 0.9 * a.bound_risk
                    + 0.25 * (a.compaction_cost / max_cost)
                    + 0.15 * (a.path_length_m / max_len)
                )
                if violation <= 1e-9:
                    target += 0.4
                rows.append(x)
                targets.append(target)
                field_ids.append(field_id)
        except Exception as exc:
            print(f"  skip {Path(wkt).stem}: {exc}")

    if not rows:
        raise RuntimeError("No training samples collected for informed sampler.")
    return np.vstack(rows), np.array(targets, dtype=float), np.array(field_ids, dtype=int)


def _field_split(field_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    fields = np.unique(field_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(fields)
    n_val = max(1, int(np.ceil(len(fields) * 0.2)))
    val_fields = set(fields[:n_val].tolist())
    val_mask = np.array([fid in val_fields for fid in field_ids], dtype=bool)
    return ~val_mask, val_mask


def _train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple:
    model = default_model(seed)
    model.x_mean = x_train.mean(axis=0)
    model.x_std = x_train.std(axis=0) + 1e-6
    xt = (x_train - model.x_mean) / model.x_std
    xv = (x_val - model.x_mean) / model.x_std

    rng = np.random.default_rng(seed)
    lr = 0.006
    weight_decay = 1e-4
    batch_size = min(64, max(8, len(xt)))
    epochs = 180

    for epoch in range(epochs):
        order = rng.permutation(len(xt))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = xt[idx]
            yb = y_train[idx].reshape(-1, 1)

            z1 = xb @ model.w1 + model.b1
            h1 = np.tanh(z1)
            z2 = h1 @ model.w2 + model.b2
            h2 = np.tanh(z2)
            pred = h2 @ model.w3 + model.b3
            err = pred - yb
            losses.append(float(np.mean(err**2)))

            scale = 2.0 / max(len(xb), 1)
            g_pred = scale * err
            gw3 = h2.T @ g_pred + weight_decay * model.w3
            gb3 = g_pred.sum(axis=0)
            gh2 = g_pred @ model.w3.T
            gz2 = gh2 * (1.0 - h2**2)
            gw2 = h1.T @ gz2 + weight_decay * model.w2
            gb2 = gz2.sum(axis=0)
            gh1 = gz2 @ model.w2.T
            gz1 = gh1 * (1.0 - h1**2)
            gw1 = xb.T @ gz1 + weight_decay * model.w1
            gb1 = gz1.sum(axis=0)

            model.w3 -= lr * gw3
            model.b3 -= lr * gb3
            model.w2 -= lr * gw2
            model.b2 -= lr * gb2
            model.w1 -= lr * gw1
            model.b1 -= lr * gb1

        if (epoch + 1) % 30 == 0:
            val_pred = model.predict(x_val)
            val_mse = float(np.mean((val_pred - y_val) ** 2))
            print(
                f"  epoch {epoch + 1:3d}/{epochs}: "
                f"train_mse={np.mean(losses):.4f}, val_mse={val_mse:.4f}"
            )

    model.trained = True
    model.val_mse = float(np.mean((model.predict(x_val) - y_val) ** 2))
    return model


def train_sampler(config: AppConfig, project_root: Path) -> int:
    x, y, field_ids = _collect_dataset(config, project_root)
    train_mask, val_mask = _field_split(field_ids, config.seed)
    if not np.any(train_mask) or not np.any(val_mask):
        raise RuntimeError("Train/validation split failed.")

    print(
        f"Training neural sampler: {train_mask.sum()} train samples, "
        f"{val_mask.sum()} validation samples"
    )
    model = _train_mlp(x[train_mask], y[train_mask], x[val_mask], y[val_mask], config.seed)

    val_pred = model.predict(x[val_mask])
    model.val_top1_acc = _field_top1_accuracy(
        val_pred, y[val_mask], field_ids[val_mask]
    )
    print(f"Validation mse={model.val_mse:.4f}, field_top1_acc={model.val_top1_acc:.1%}")

    out = project_root / (config.informed_sampling.model_path or "models/informed_sampler.npz")
    save_model(model, out)
    print(f"  saved {out}")
    return 0


def _field_top1_accuracy(pred: np.ndarray, target: np.ndarray, field_ids: np.ndarray) -> float:
    hits = 0
    total = 0
    for fid in np.unique(field_ids):
        idx = np.where(field_ids == fid)[0]
        if len(idx) == 0:
            continue
        hits += int(idx[int(np.argmax(pred[idx]))] == idx[int(np.argmax(target[idx]))])
        total += 1
    return hits / max(total, 1)


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "nr_ccp_full.yaml")
    raise SystemExit(train_sampler(cfg, ROOT))
