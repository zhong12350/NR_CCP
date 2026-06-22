"""Informed angle sampling for NR-CCP (lightweight learned prior)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config_loader import InformedSamplingConfig
from src.fields import RiskField
from src.geometry import FieldGrid


@dataclass
class SamplerModel:
    """Tiny MLP: field/angle features -> angle quality score."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)
        return (h @ self.w2 + self.b2).ravel()


def default_model(seed: int = 42) -> SamplerModel:
    rng = np.random.default_rng(seed)
    return SamplerModel(
        w1=rng.normal(0, 0.25, size=(8, 16)),
        b1=rng.normal(0, 0.05, size=(16,)),
        w2=rng.normal(0, 0.25, size=(16, 1)),
        b2=np.array([0.0]),
    )


def load_model(path: str | Path | None, seed: int = 42) -> SamplerModel:
    if path is None:
        return default_model(seed)
    p = Path(path)
    if not p.exists():
        return default_model(seed)
    data = np.load(p)
    return SamplerModel(
        w1=data["w1"],
        b1=data["b1"],
        w2=data["w2"],
        b2=data["b2"],
    )


def save_model(model: SamplerModel, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, w1=model.w1, b1=model.b1, w2=model.w2, b2=model.b2)


class InformedAngleSampler:
    """Score swath angles using field/risk features + learned prior."""

    def __init__(self, cfg: InformedSamplingConfig, seed: int = 42):
        self.cfg = cfg
        self.model = load_model(cfg.model_path, seed=seed)

    def field_features(self, grid: FieldGrid, risk: RiskField) -> np.ndarray:
        geom = grid.geometry
        inner_vals = risk.values[grid.inner_mask]
        mean_r = float(inner_vals.mean()) if inner_vals.size else 0.0
        std_r = float(inner_vals.std()) if inner_vals.size else 0.0
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
                mean_r,
                std_r,
                float(grid.headland_mask.mean()),
            ],
            dtype=float,
        )

    def angle_proxy_risk(
        self,
        grid: FieldGrid,
        risk: RiskField,
        angle_deg: float,
        swath_width_m: float,
    ) -> float:
        from src.candidates import _generate_swath_lines

        swaths = _generate_swath_lines(grid, angle_deg, swath_width_m)
        if not swaths:
            return 1.0
        half = swath_width_m / 2.0
        samples: list[float] = []
        for swath in swaths:
            n = max(2, int(swath.length / 4.0) + 1)
            for t in np.linspace(0.0, 1.0, n):
                p = swath.interpolate(t, normalized=True)
                samples.append(risk.sample(p.x, p.y, grid))
        return float(np.mean(samples)) if samples else 1.0

    def score_angle(
        self,
        grid: FieldGrid,
        risk: RiskField,
        angle_deg: float,
        swath_width_m: float,
    ) -> float:
        ff = self.field_features(grid, risk)
        proxy = self.angle_proxy_risk(grid, risk, angle_deg, swath_width_m)
        rad = np.deg2rad(angle_deg)
        x = np.concatenate(
            [
                ff,
                np.array(
                    [
                        np.sin(rad),
                        np.cos(rad),
                        np.sin(2 * rad),
                        np.cos(2 * rad),
                        proxy,
                        proxy * ff[5],
                    ],
                    dtype=float,
                ),
            ]
        )
        if x.shape[0] < self.model.w1.shape[0]:
            x = np.pad(x, (0, self.model.w1.shape[0] - x.shape[0]))
        else:
            x = x[: self.model.w1.shape[0]]
        learned = float(self.model.predict(x.reshape(1, -1))[0])
        heuristic = -(1.2 * proxy + 0.15 * abs(np.sin(np.deg2rad(angle_deg - 90.0))))
        return (1.0 - self.cfg.learned_weight) * heuristic + self.cfg.learned_weight * learned

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
