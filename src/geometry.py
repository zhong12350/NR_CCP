"""Field geometry and coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config_loader import FieldConfig


@dataclass(frozen=True)
class FieldGrid:
    """Rectangular farmland discretized on a regular grid."""

    width_m: float
    height_m: float
    cell_size_m: float
    obstacle_mask: np.ndarray  # shape (ny, nx), True = obstacle

    @property
    def nx(self) -> int:
        return int(self.obstacle_mask.shape[1])

    @property
    def ny(self) -> int:
        return int(self.obstacle_mask.shape[0])

    @property
    def x_coords(self) -> np.ndarray:
        return np.arange(self.nx) * self.cell_size_m + self.cell_size_m / 2

    @property
    def y_coords(self) -> np.ndarray:
        return np.arange(self.ny) * self.cell_size_m + self.cell_size_m / 2

    def is_valid_world(self, x: float, y: float) -> bool:
        if not (0.0 <= x <= self.width_m and 0.0 <= y <= self.height_m):
            return False
        ix, iy = self.world_to_index(x, y)
        return not self.obstacle_mask[iy, ix]

    def world_to_index(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.clip(x / self.cell_size_m, 0, self.nx - 1))
        iy = int(np.clip(y / self.cell_size_m, 0, self.ny - 1))
        return ix, iy

    def index_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        return self.x_coords[ix], self.y_coords[iy]

    def free_cells(self) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                if not self.obstacle_mask[iy, ix]:
                    points.append(self.index_to_world(ix, iy))
        return points


def build_field_grid(field_cfg: FieldConfig) -> FieldGrid:
    """Construct a field grid from configuration."""
    nx = max(1, int(np.ceil(field_cfg.width_m / field_cfg.cell_size_m)))
    ny = max(1, int(np.ceil(field_cfg.height_m / field_cfg.cell_size_m)))
    obstacle_mask = np.zeros((ny, nx), dtype=bool)

    for obs in field_cfg.obstacles:
        if len(obs) != 4:
            continue
        x_min, y_min, x_max, y_max = obs
        ix0 = int(np.floor(x_min / field_cfg.cell_size_m))
        ix1 = int(np.ceil(x_max / field_cfg.cell_size_m))
        iy0 = int(np.floor(y_min / field_cfg.cell_size_m))
        iy1 = int(np.ceil(y_max / field_cfg.cell_size_m))
        ix0 = max(0, ix0)
        iy0 = max(0, iy0)
        ix1 = min(nx, ix1)
        iy1 = min(ny, iy1)
        obstacle_mask[iy0:iy1, ix0:ix1] = True

    return FieldGrid(
        width_m=field_cfg.width_m,
        height_m=field_cfg.height_m,
        cell_size_m=field_cfg.cell_size_m,
        obstacle_mask=obstacle_mask,
    )
