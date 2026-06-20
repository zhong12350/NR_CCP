"""Risk field generation (compaction / soil damage proxy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config_loader import GaussianSpec, RiskFieldConfig
from src.geometry import FieldGrid


@dataclass
class RiskField:
    """2D risk intensity map aligned with the field grid."""

    values: np.ndarray  # shape (ny, nx), in [0, 1] after normalization

    def sample(self, x: float, y: float, grid: FieldGrid) -> float:
        ix, iy = grid.world_to_index(x, y)
        return float(self.values[iy, ix])

    def sample_path(self, waypoints: list[tuple[float, float]], grid: FieldGrid) -> np.ndarray:
        return np.array([self.sample(x, y, grid) for x, y in waypoints], dtype=float)


def build_risk_field(grid: FieldGrid, cfg: RiskFieldConfig) -> RiskField:
    """Build a compaction risk field from Gaussian hotspots."""
    xx, yy = np.meshgrid(grid.x_coords, grid.y_coords)
    risk = np.zeros((grid.ny, grid.nx), dtype=float)

    for spec in cfg.gaussians:
        cx, cy = spec.center
        sx, sy = spec.sigma
        gaussian = spec.amplitude * np.exp(
            -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
        )
        risk += gaussian

    risk[grid.obstacle_mask] = 0.0

    if cfg.normalize and risk.max() > 0:
        risk = risk / risk.max()

    return RiskField(values=risk)


def strip_mean_risk(
    risk: RiskField,
    grid: FieldGrid,
    x_center: float,
    half_width: float,
) -> float:
    """Mean risk over cells in a vertical strip centered at x_center."""
    mask = (grid.x_coords >= x_center - half_width) & (
        grid.x_coords <= x_center + half_width
    )
    if not np.any(mask):
        return 0.0
    strip = risk.values[:, mask]
    free = ~grid.obstacle_mask[:, mask]
    if not np.any(free):
        return 0.0
    return float(strip[free].mean())
