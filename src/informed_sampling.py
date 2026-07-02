"""Neural informed angle sampling for NR-CCP.

The sampler uses a trained MLP prior over field/risk/angle features. Unlike
the original prototype, the full feature vector is preserved, standardized,
and saved with validation metadata. When no trained model exists, the module
falls back to a deterministic uncertainty-aware heuristic and reports that via
the model metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config_loader import InformedSamplingConfig
from src.fields import RiskField
from src.geometry import FieldGrid

FEATURE_DIM = 21


@dataclass
class SamplerModel:
    """Two-hidden-layer MLP: field/risk/angle features -> angle quality score."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    x_mean: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIM))
    x_std: np.ndarray = field(default_factory=lambda: np.ones(FEATURE_DIM))
    trained: bool = False
    val_mse: float = float("nan")
    val_top1_acc: float = float("nan")

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.x_mean) / np.maximum(self.x_std, 1e-6)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        return (h2 @ self.w3 + self.b3).ravel()


def default_model(seed: int = 42) -> SamplerModel:
    rng = np.random.default_rng(seed)
    return SamplerModel(
        w1=rng.normal(0, 0.08, size=(FEATURE_DIM, 32)),
        b1=np.zeros(32),
        w2=rng.normal(0, 0.08, size=(32, 16)),
        b2=np.zeros(16),
        w3=rng.normal(0, 0.08, size=(16, 1)),
        b3=np.array([0.0]),
        trained=False,
    )


def load_model(path: str | Path | None, seed: int = 42) -> SamplerModel:
    if path is None:
        return default_model(seed)
    p = Path(path)
    if not p.exists():
        return default_model(seed)
    data = np.load(p)
    if not {"w1", "b1", "w2", "b2", "w3", "b3"}.issubset(set(data.files)):
        return default_model(seed)
    w1 = data["w1"]
    if w1.shape[0] != FEATURE_DIM:
        return default_model(seed)
    return SamplerModel(
        w1=w1,
        b1=data["b1"],
        w2=data["w2"],
        b2=data["b2"],
        w3=data["w3"],
        b3=data["b3"],
        x_mean=data["x_mean"] if "x_mean" in data else np.zeros(FEATURE_DIM),
        x_std=data["x_std"] if "x_std" in data else np.ones(FEATURE_DIM),
        trained=bool(data["trained"]) if "trained" in data else True,
        val_mse=float(data["val_mse"]) if "val_mse" in data else float("nan"),
        val_top1_acc=float(data["val_top1_acc"]) if "val_top1_acc" in data else float("nan"),
    )


def save_model(model: SamplerModel, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        w3=model.w3,
        b3=model.b3,
        x_mean=model.x_mean,
        x_std=model.x_std,
        trained=np.array(model.trained),
        val_mse=np.array(model.val_mse),
        val_top1_acc=np.array(model.val_top1_acc),
    )


def _safe_inner_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask]
    return vals if vals.size else np.array([0.0], dtype=float)


class InformedAngleSampler:
    """Score swath angles using field/risk features + learned prior."""

    def __init__(self, cfg: InformedSamplingConfig, seed: int = 42):
        self.cfg = cfg
        self.model = load_model(cfg.model_path, seed=seed)

    def field_features(self, grid: FieldGrid, risk: RiskField) -> np.ndarray:
        geom = grid.geometry
        inner_vals = _safe_inner_values(risk.values, grid.inner_mask)
        unc_vals = _safe_inner_values(risk.uncertainty, grid.inner_mask)
        minx, miny, maxx, maxy = geom.inner.bounds
        w = max(maxx - minx, 1.0)
        h = max(maxy - miny, 1.0)
        return np.array(
            [
                1.0,
                np.log1p(geom.area_m2) / 12.0,
                geom.aspect_ratio / 5.0,
                w / 500.0,
                h / 500.0,
                float(inner_vals.mean()),
                float(inner_vals.std()),
                float(np.quantile(inner_vals, 0.90)),
                float(unc_vals.mean()),
                float(unc_vals.std()),
                float(np.quantile(unc_vals, 0.90)),
                float(grid.headland_mask.mean()),
                float(geom.outer.length / max(geom.area_m2, 1.0)),
            ],
            dtype=float,
        )

    def angle_proxy_features(
        self,
        grid: FieldGrid,
        risk: RiskField,
        angle_deg: float,
        swath_width_m: float,
    ) -> tuple[float, float, float, float]:
        from src.candidates import _generate_swath_lines

        swaths = _generate_swath_lines(grid, angle_deg, swath_width_m)
        if not swaths:
            return 1.0, 1.0, 0.0, 0.0
        risk_samples: list[float] = []
        unc_samples: list[float] = []
        total_len = 0.0
        for swath in swaths:
            total_len += swath.length
            n = max(2, int(swath.length / 4.0) + 1)
            for t in np.linspace(0.0, 1.0, n):
                p = swath.interpolate(t, normalized=True)
                risk_samples.append(risk.sample(p.x, p.y, grid))
                unc_samples.append(float(risk.sample_uncertainty_many(np.array([p.x]), np.array([p.y]), grid)[0]))
        proxy_risk = float(np.mean(risk_samples)) if risk_samples else 1.0
        proxy_unc = float(np.mean(unc_samples)) if unc_samples else 1.0
        return proxy_risk, proxy_unc, len(swaths) / 80.0, total_len / 5000.0

    def angle_proxy_risk(
        self,
        grid: FieldGrid,
        risk: RiskField,
        angle_deg: float,
        swath_width_m: float,
    ) -> float:
        return self.angle_proxy_features(grid, risk, angle_deg, swath_width_m)[0]

    def feature_vector(
        self,
        grid: FieldGrid,
        risk: RiskField,
        angle_deg: float,
        swath_width_m: float,
    ) -> np.ndarray:
        ff = self.field_features(grid, risk)
        proxy, proxy_unc, swath_count, total_len = self.angle_proxy_features(
            grid, risk, angle_deg, swath_width_m
        )
        rad = np.deg2rad(angle_deg)
        angle_features = np.array(
            [
                np.sin(rad),
                np.cos(rad),
                np.sin(2 * rad),
                np.cos(2 * rad),
                proxy,
                proxy_unc,
                swath_count,
                total_len,
            ],
            dtype=float,
        )
        x = np.concatenate([ff, angle_features])
        if x.shape[0] != FEATURE_DIM:
            raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {x.shape[0]}")
        return x

    def score_angle(
        self,
        grid: FieldGrid,
        risk: RiskField,
        angle_deg: float,
        swath_width_m: float,
    ) -> float:
        x = self.feature_vector(grid, risk, angle_deg, swath_width_m)
        proxy = x[-4]
        proxy_unc = x[-3]
        learned = float(self.model.predict(x.reshape(1, -1))[0])
        heuristic = -(
            1.15 * proxy
            + 0.65 * proxy_unc
            + 0.10 * x[-2]
            + 0.10 * abs(np.sin(np.deg2rad(angle_deg - 90.0)))
        )
        learned_weight = self.cfg.learned_weight if self.model.trained else 0.15
        return (1.0 - learned_weight) * heuristic + learned_weight * learned

    def propose_angles(
        self,
        grid: FieldGrid,
        risk: RiskField,
        swath_width_m: float,
    ) -> list[float]:
        """Return prioritized angle list for risk-aware search."""
        coarse = np.arange(0.0, 180.0, self.cfg.coarse_step_deg)
        scored = [
            (self.score_angle(grid, risk, float(a), swath_width_m), float(a))
            for a in coarse
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        chosen: list[float] = []
        for _, angle in scored[: self.cfg.top_k_coarse]:
            chosen.append(angle)
            for d in (-self.cfg.fine_step_deg, 0.0, self.cfg.fine_step_deg):
                refined = (angle + d) % 180.0
                if refined not in chosen:
                    chosen.append(refined)
        if self.cfg.include_principal_axes:
            for axis in (0.0, 90.0):
                if axis not in chosen:
                    chosen.append(axis)
        return sorted(set(round(a, 4) for a in chosen))
