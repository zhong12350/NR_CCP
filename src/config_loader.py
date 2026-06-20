"""Load and validate YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FieldConfig:
    width_m: float = 100.0
    height_m: float = 60.0
    cell_size_m: float = 1.0
    obstacles: list[list[float]] = dc_field(default_factory=list)


@dataclass
class GaussianSpec:
    center: tuple[float, float]
    sigma: tuple[float, float]
    amplitude: float


@dataclass
class RiskFieldConfig:
    gaussians: list[GaussianSpec] = dc_field(default_factory=list)
    normalize: bool = True


@dataclass
class PlannerConfig:
    strip_width_m: float = 4.0
    waypoint_spacing_m: float = 2.0
    start: tuple[float, float] = (2.0, 2.0)


@dataclass
class OutputConfig:
    figures_dir: Path = Path("outputs/figures")
    results_dir: Path = Path("outputs/results")
    dpi: int = 150


@dataclass
class AppConfig:
    seed: int = 42
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    risk_field: RiskFieldConfig = dc_field(default_factory=RiskFieldConfig)
    planner: PlannerConfig = dc_field(default_factory=PlannerConfig)
    methods: list[str] = dc_field(default_factory=list)
    output: OutputConfig = dc_field(default_factory=OutputConfig)


def _parse_gaussian(raw: dict[str, Any]) -> GaussianSpec:
    return GaussianSpec(
        center=(float(raw["center"][0]), float(raw["center"][1])),
        sigma=(float(raw["sigma"][0]), float(raw["sigma"][1])),
        amplitude=float(raw["amplitude"]),
    )


def load_config(path: str | Path) -> AppConfig:
    """Load application config from a YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    gaussians = [_parse_gaussian(g) for g in raw.get("risk_field", {}).get("gaussians", [])]
    field_raw = raw.get("field", {})
    planner_raw = raw.get("planner", {})
    output_raw = raw.get("output", {})

    return AppConfig(
        seed=int(raw.get("seed", 42)),
        field=FieldConfig(
            width_m=float(field_raw.get("width_m", 100.0)),
            height_m=float(field_raw.get("height_m", 60.0)),
            cell_size_m=float(field_raw.get("cell_size_m", 1.0)),
            obstacles=[list(o) for o in field_raw.get("obstacles", [])],
        ),
        risk_field=RiskFieldConfig(
            gaussians=gaussians,
            normalize=bool(raw.get("risk_field", {}).get("normalize", True)),
        ),
        planner=PlannerConfig(
            strip_width_m=float(planner_raw.get("strip_width_m", 4.0)),
            waypoint_spacing_m=float(planner_raw.get("waypoint_spacing_m", 2.0)),
            start=(
                float(planner_raw.get("start", [2.0, 2.0])[0]),
                float(planner_raw.get("start", [2.0, 2.0])[1]),
            ),
        ),
        methods=list(raw.get("methods", [])),
        output=OutputConfig(
            figures_dir=Path(output_raw.get("figures_dir", "outputs/figures")),
            results_dir=Path(output_raw.get("results_dir", "outputs/results")),
            dpi=int(output_raw.get("dpi", 150)),
        ),
    )
