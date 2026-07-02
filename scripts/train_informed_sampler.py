#!/usr/bin/env python3
"""Train lightweight informed angle sampler (fast, stable)."""

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


def _feature_vector(sampler: InformedAngleSampler, grid, risk, angle: float, swath_w: float) -> np.ndarray:
    ff = sampler.field_features(grid, risk)
    proxy = sampler.angle_proxy_risk(grid, risk, angle, swath_w)
    rad = np.deg2rad(angle)
    x = np.concatenate(
        [
            ff,
            np.array(
                [np.sin(rad), np.cos(rad), np.sin(2 * rad), np.cos(2 * rad), proxy, proxy * ff[5]]
            ),
        ]
    )
    if x.shape[0] < 8:
        x = np.pad(x, (0, 8 - x.shape[0]))
    return x[:8]


def train_sampler(config: AppConfig, project_root: Path) -> int:
    pattern = str(project_root / config.batch.field_glob)
    wkt_files = sorted(glob(pattern))[:12]
    model = default_model(config.seed)
    lr = 0.01
    epochs = 5

    print(f"Training informed sampler on {len(wkt_files)} fields...")
    for epoch in range(epochs):
        loss_sum = 0.0
        n = 0
        for wkt in wkt_files:
            try:
                grid = load_field_from_wkt(
                    wkt,
                    config.headland.width_m,
                    auto_cell_size=config.field.auto_cell_size,
                )
                risk = build_risk_field(grid, config.risk_field, config.planner)
                full = enumerate_candidates(grid, config.planner)
                assess = assess_all_candidates(
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
                if not assess:
                    continue
                best = min(assess, key=lambda a: a.mean_risk + 0.001 * a.path_length_m)
                sampler = InformedAngleSampler(config.informed_sampling, config.seed)

                min_len = min(a.path_length_m for a in assess)
                for angle in np.arange(0, 180, 15):
                    x = _feature_vector(
                        sampler, grid, risk, float(angle), config.planner.swath_width_m
                    )
                    proxy = sampler.angle_proxy_risk(
                        grid, risk, float(angle), config.planner.swath_width_m
                    )
                    # Target: low violation proxy + low path-length cost angle.
                    target = -(proxy + 0.001 * min_len)
                    if abs(angle - best.angle_deg) < 5:
                        target += 0.5
                    h = np.tanh(np.clip(x @ model.w1 + model.b1, -10, 10))
                    pred = float((h @ model.w2 + model.b2).item())
                    err = np.clip(pred - target, -2.0, 2.0)
                    loss_sum += err * err
                    n += 1

                    grad_out = 2 * err
                    model.w2 += lr * grad_out * h.reshape(-1, 1)
                    model.b2[0] += lr * grad_out
                    dh = np.clip(grad_out * model.w2.ravel(), -1.0, 1.0)
                    dtanh = (1 - h**2) * dh
                    model.w1 += lr * np.outer(x, dtanh)
                    model.b1 += lr * dtanh
            except Exception as exc:
                print(f"  skip {Path(wkt).stem}: {exc}")

        print(f"  epoch {epoch + 1}/{epochs} mse={loss_sum / max(n, 1):.4f}")

    out = project_root / (config.informed_sampling.model_path or "models/informed_sampler.npz")
    save_model(model, out)
    print(f"  saved {out}")
    return 0


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "nr_ccp_full.yaml")
    raise SystemExit(train_sampler(cfg, ROOT))
