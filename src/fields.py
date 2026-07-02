"""Planning risk proxy: headland + hotspots + pass-count accumulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point

from src.config_loader import GaussianSpec, PlannerConfig, RiskFieldConfig
from src.geometry import FieldGrid


@dataclass
class RiskField:
    """Static planning risk map R(x) in [0, 1] with decomposed layers."""

    values: np.ndarray
    headland_layer: np.ndarray
    hotspot_layer: np.ndarray
    pass_count_layer: np.ndarray
    pass_count: np.ndarray | None = None

    def sample(self, x: float, y: float, grid: FieldGrid) -> float:
        ix, iy = grid.world_to_index(x, y)
        return float(self.values[iy, ix])

    def sample_many(self, xs: np.ndarray, ys: np.ndarray, grid: FieldGrid) -> np.ndarray:
        ixs = ((xs - grid.origin_x) / grid.cell_size_m).astype(int)
        iys = ((ys - grid.origin_y) / grid.cell_size_m).astype(int)
        ixs = np.clip(ixs, 0, grid.nx - 1)
        iys = np.clip(iys, 0, grid.ny - 1)
        return self.values[iys, ixs]

    def sample_layer_many(
        self, layer: np.ndarray, xs: np.ndarray, ys: np.ndarray, grid: FieldGrid
    ) -> np.ndarray:
        ixs = ((xs - grid.origin_x) / grid.cell_size_m).astype(int)
        iys = ((ys - grid.origin_y) / grid.cell_size_m).astype(int)
        ixs = np.clip(ixs, 0, grid.nx - 1)
        iys = np.clip(iys, 0, grid.ny - 1)
        return layer[iys, ixs]


def _auto_gaussians(grid: FieldGrid, n: int = 2) -> list[GaussianSpec]:
    inner = grid.geometry.inner
    minx, miny, maxx, maxy = inner.bounds
    w = max(maxx - minx, 1.0)
    h = max(maxy - miny, 1.0)
    cx, cy = inner.centroid.x, inner.centroid.y
    specs: list[GaussianSpec] = []
    offsets = [(0.0, 0.0), (0.22, -0.18), (-0.2, 0.15)]
    for i in range(min(n, len(offsets))):
        ox, oy = offsets[i]
        center = (cx + ox * w, cy + oy * h)
        if not inner.contains(Point(center)):
            center = (cx, cy)
        sigma = (max(w * 0.12, 6.0), max(h * 0.12, 6.0))
        specs.append(GaussianSpec(center=center, sigma=sigma, amplitude=0.45 - 0.08 * i))
    return specs


def estimate_pass_count(grid: FieldGrid, angle_deg: float, swath_width_m: float) -> np.ndarray:
    from src.candidates import _generate_swath_lines

    counts = np.zeros((grid.ny, grid.nx), dtype=float)
    half = swath_width_m / 2.0
    swaths = _generate_swath_lines(grid, angle_deg, swath_width_m)
    yy, xx = np.meshgrid(grid.y_coords, grid.x_coords, indexing="ij")

    for swath in swaths:
        for i in range(len(swath.coords) - 1):
            x0, y0 = swath.coords[i]
            x1, y1 = swath.coords[i + 1]
            seg_len = float(np.hypot(x1 - x0, y1 - y0))
            n = max(2, int(seg_len / grid.cell_size_m) + 1)
            xs = np.linspace(x0, x1, n)
            ys = np.linspace(y0, y1, n)
            for x, y in zip(xs, ys):
                dist_sq = (xx - x) ** 2 + (yy - y) ** 2
                counts[dist_sq <= half**2] += 1.0

    counts[~grid.inner_mask] = 0.0
    return counts


def build_pass_count_layer(
    grid: FieldGrid,
    planner_cfg: PlannerConfig,
    cfg: RiskFieldConfig,
) -> np.ndarray:
    if not cfg.use_pass_count:
        return np.zeros((grid.ny, grid.nx), dtype=float)

    angles = cfg.pass_count_angles or [0.0, 45.0, 90.0, 135.0]
    combined = np.zeros((grid.ny, grid.nx), dtype=float)
    for angle in angles:
        combined = np.maximum(
            combined,
            estimate_pass_count(grid, angle, planner_cfg.swath_width_m),
        )
    if combined.max() > 0:
        combined = combined / combined.max()
    return combined


def build_risk_field(
    grid: FieldGrid,
    cfg: RiskFieldConfig,
    planner_cfg: PlannerConfig | None = None,
) -> RiskField:
    """Build decomposed static planning risk proxy."""
    headland_layer = np.zeros((grid.ny, grid.nx), dtype=float)
    hotspot_layer = np.zeros((grid.ny, grid.nx), dtype=float)
    pass_count_layer = np.zeros((grid.ny, grid.nx), dtype=float)
    risk = np.full((grid.ny, grid.nx), cfg.inner_base, dtype=float)
    geom = grid.geometry
    decay = max(cfg.headland_decay_m, 1e-3)
    xx, yy = np.meshgrid(grid.x_coords, grid.y_coords)

    for iy in range(grid.ny):
        for ix in range(grid.nx):
            if not grid.outer_mask[iy, ix]:
                risk[iy, ix] = 0.0
                continue
            x, y = grid.index_to_world(ix, iy)
            pt = Point(x, y)
            if grid.headland_mask[iy, ix]:
                headland_layer[iy, ix] = cfg.headland_base
                risk[iy, ix] = cfg.headland_base
            else:
                dist_headland = pt.distance(geom.inner.boundary)
                boost = (cfg.headland_base - cfg.inner_base) * np.exp(
                    -dist_headland / decay
                )
                headland_layer[iy, ix] = cfg.inner_base + boost
                risk[iy, ix] = cfg.inner_base + boost

    gaussians = list(cfg.gaussians)
    if cfg.auto_hotspots and not gaussians:
        gaussians = _auto_gaussians(grid, cfg.auto_hotspot_count)

    for spec in gaussians:
        cx, cy = spec.center
        sx, sy = spec.sigma
        hotspot = spec.amplitude * np.exp(
            -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
        )
        hotspot_layer += hotspot
        risk += hotspot

    if planner_cfg is not None and cfg.use_pass_count:
        pass_count_layer = build_pass_count_layer(grid, planner_cfg, cfg)
        risk += cfg.pass_count_weight * pass_count_layer

    risk[~grid.outer_mask] = 0.0
    headland_layer[~grid.outer_mask] = 0.0
    hotspot_layer[~grid.outer_mask] = 0.0
    pass_count_layer[~grid.outer_mask] = 0.0

    if cfg.normalize and risk.max() > 0:
        scale = risk.max()
        risk = risk / scale
        headland_layer = headland_layer / scale
        hotspot_layer = hotspot_layer / scale
        pass_count_layer = pass_count_layer / scale

    return RiskField(
        values=risk,
        headland_layer=headland_layer,
        hotspot_layer=hotspot_layer,
        pass_count_layer=pass_count_layer,
        pass_count=pass_count_layer if cfg.use_pass_count else None,
    )
