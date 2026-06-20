"""Coverage path planners for NR-CCP demo."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from src.config_loader import PlannerConfig
from src.fields import RiskField, strip_mean_risk
from src.geometry import FieldGrid


@dataclass(frozen=True)
class PlanResult:
    """Output of a coverage planner."""

    method: str
    waypoints: list[tuple[float, float]]
    strip_centers: list[float]


class BaseCoveragePlanner(ABC):
    """Common strip generation for boustrophedon-style coverage."""

    def __init__(self, grid: FieldGrid, planner_cfg: PlannerConfig):
        self.grid = grid
        self.cfg = planner_cfg

    @property
    @abstractmethod
    def method_name(self) -> str:
        ...

    def plan(self, risk: RiskField) -> PlanResult:
        strips = self._build_strips()
        order = self._order_strips(strips, risk)
        waypoints = self._connect_strips(order, strips)
        return PlanResult(
            method=self.method_name,
            waypoints=waypoints,
            strip_centers=[s["x_center"] for s in order],
        )

    def _build_strips(self) -> list[dict]:
        """Vertical strips spanning the field height."""
        half = self.cfg.strip_width_m / 2.0
        spacing = self.cfg.waypoint_spacing_m
        x = half
        strips: list[dict] = []

        while x <= self.grid.width_m - half + 1e-6:
            y_points = np.arange(half, self.grid.height_m - half + spacing, spacing)
            if len(y_points) == 0:
                y_points = np.array([self.grid.height_m / 2])
            strips.append(
                {
                    "x_center": float(x),
                    "y_points": y_points.astype(float),
                    "half_width": half,
                }
            )
            x += self.cfg.strip_width_m

        return strips

    @abstractmethod
    def _order_strips(self, strips: list[dict], risk: RiskField) -> list[dict]:
        ...

    def _connect_strips(
        self, ordered_strips: list[dict], all_strips: list[dict]
    ) -> list[tuple[float, float]]:
        """Snake through ordered strips; alternate up/down each strip."""
        if not ordered_strips:
            return [self.cfg.start]

        waypoints: list[tuple[float, float]] = [self.cfg.start]
        current = self.cfg.start
        going_up = True

        for strip in ordered_strips:
            x = strip["x_center"]
            y_list = list(strip["y_points"])
            if not going_up:
                y_list = list(reversed(y_list))

            entry = (x, float(y_list[0]))
            waypoints.extend(self._bridge(current, entry))
            for y in y_list:
                waypoints.append((x, float(y)))
            current = (x, float(y_list[-1]))
            going_up = not going_up

        return waypoints

    def _bridge(
        self, a: tuple[float, float], b: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Simple axis-aligned connector between strip endpoints."""
        if np.allclose(a, b):
            return []
        return [(b[0], a[1]), b]

    def _bridge_candidates(
        self, a: tuple[float, float], b: tuple[float, float]
    ) -> list[list[tuple[float, float]]]:
        if np.allclose(a, b):
            return [[]]
        direct = [(b[0], a[1]), b]
        margin = self.cfg.waypoint_spacing_m
        y_bottom = margin
        y_top = self.grid.height_m - margin
        via_bottom = [(a[0], y_bottom), (b[0], y_bottom), b]
        via_top = [(a[0], y_top), (b[0], y_top), b]
        return [direct, via_bottom, via_top]


def _bridge_risk_cost(
    bridge: list[tuple[float, float]],
    start: tuple[float, float],
    risk: RiskField,
    grid: FieldGrid,
) -> float:
    if not bridge:
        return 0.0
    pts = [start] + bridge
    cost = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        length = abs(x1 - x0) + abs(y1 - y0)
        mid_r = risk.sample((x0 + x1) / 2, (y0 + y1) / 2, grid)
        cost += length * (1.0 + 4.0 * mid_r)
    return cost


class UniformBoustrophedonPlanner(BaseCoveragePlanner):
    """Standard left-to-right boustrophedon; ignores risk."""

    @property
    def method_name(self) -> str:
        return "uniform_boustrophedon"

    def _order_strips(self, strips: list[dict], risk: RiskField) -> list[dict]:
        return strips


class RiskAwareReorderingPlanner(BaseCoveragePlanner):
    """Same strip order as uniform; detour transitions through lower-risk corridors."""

    def __init__(self, grid: FieldGrid, planner_cfg: PlannerConfig):
        super().__init__(grid, planner_cfg)
        self._risk: RiskField | None = None

    @property
    def method_name(self) -> str:
        return "risk_aware_reordering"

    def plan(self, risk: RiskField) -> PlanResult:
        self._risk = risk
        try:
            return super().plan(risk)
        finally:
            self._risk = None

    def _order_strips(self, strips: list[dict], risk: RiskField) -> list[dict]:
        return strips

    def _bridge(
        self, a: tuple[float, float], b: tuple[float, float]
    ) -> list[tuple[float, float]]:
        if self._risk is None:
            return super()._bridge(a, b)
        direct = [(b[0], a[1]), b]
        mid_r = self._risk.sample((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, self.grid)
        if mid_r < 0.35:
            return direct
        return min(
            self._bridge_candidates(a, b),
            key=lambda cand: _bridge_risk_cost(cand, a, self._risk, self.grid),
        )


class InformedStripSelectionPlanner(BaseCoveragePlanner):
    """Visit higher-risk strips earlier (informed prior on compaction hotspots)."""

    @property
    def method_name(self) -> str:
        return "informed_strip_selection"

    def _order_strips(self, strips: list[dict], risk: RiskField) -> list[dict]:
        scored = []
        for strip in strips:
            mean_r = strip_mean_risk(
                risk, self.grid, strip["x_center"], strip["half_width"]
            )
            scored.append((mean_r, strip))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [s for _, s in scored]


PLANNER_REGISTRY: dict[str, type[BaseCoveragePlanner]] = {
    "uniform_boustrophedon": UniformBoustrophedonPlanner,
    "risk_aware_reordering": RiskAwareReorderingPlanner,
    "informed_strip_selection": InformedStripSelectionPlanner,
}


def create_planner(method: str, grid: FieldGrid, cfg: PlannerConfig) -> BaseCoveragePlanner:
    if method not in PLANNER_REGISTRY:
        raise ValueError(
            f"Unknown planner '{method}'. Available: {list(PLANNER_REGISTRY)}"
        )
    return PLANNER_REGISTRY[method](grid, cfg)


def plan_all(
    methods: list[str],
    grid: FieldGrid,
    risk: RiskField,
    planner_cfg: PlannerConfig,
) -> list[PlanResult]:
    results: list[PlanResult] = []
    for method in methods:
        planner = create_planner(method, grid, planner_cfg)
        results.append(planner.plan(risk))
    return results
