"""Load and validate YAML configuration for RB-CCP / NR-CCP."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FieldConfig:
    wkt_path: str = "wkt/ee_field_10.wkt"
    cell_size_m: float | None = None
    auto_cell_size: bool = True


@dataclass
class HeadlandConfig:
    width_m: float = 18.0


@dataclass
class GaussianSpec:
    center: tuple[float, float]
    sigma: tuple[float, float]
    amplitude: float


@dataclass
class RiskFieldConfig:
    headland_base: float = 1.0
    inner_base: float = 0.08
    headland_decay_m: float = 6.0
    gaussians: list[GaussianSpec] = dc_field(default_factory=list)
    auto_hotspots: bool = True
    auto_hotspot_count: int = 2
    use_pass_count: bool = True
    pass_count_weight: float = 0.25
    pass_count_angles: list[float] = dc_field(default_factory=lambda: [0.0, 45.0, 90.0, 135.0])
    repeat_penalty: float = 0.35
    turning_penalty: float = 0.25
    turning_angle_deg: float = 45.0
    normalize: bool = True


@dataclass
class PlannerConfig:
    swath_width_m: float = 6.0
    angle_step_deg: float = 5.0
    min_coverage: float = 0.90
    waypoint_spacing_m: float = 2.0


@dataclass
class SelectionConfig:
    delta: float = 0.38
    lambda_weighted: float = 0.8
    beta_rb_ccp: float = 1.0
    beta_nr_ccp: float = 1.0


@dataclass
class InformedSamplingConfig:
    enabled: bool = True
    coarse_step_deg: float = 15.0
    fine_step_deg: float = 2.0
    top_k_coarse: int = 6
    include_principal_axes: bool = True
    learned_weight: float = 0.35
    model_path: str | None = "models/informed_sampler.npz"


@dataclass
class BatchConfig:
    field_glob: str = "wkt/*.wkt"
    max_fields: int = 350
    save_figures: bool = False
    progress_every: int = 10


@dataclass
class OutputConfig:
    figures_dir: Path = Path("outputs/figures")
    results_dir: Path = Path("outputs/results")
    advisor_demo_dir: Path = Path("outputs/advisor_demo")
    dpi: int = 150


@dataclass
class AppConfig:
    seed: int = 42
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    headland: HeadlandConfig = dc_field(default_factory=HeadlandConfig)
    risk_field: RiskFieldConfig = dc_field(default_factory=RiskFieldConfig)
    planner: PlannerConfig = dc_field(default_factory=PlannerConfig)
    selection: SelectionConfig = dc_field(default_factory=SelectionConfig)
    informed_sampling: InformedSamplingConfig = dc_field(default_factory=InformedSamplingConfig)
    methods: list[str] = dc_field(default_factory=list)
    batch: BatchConfig = dc_field(default_factory=BatchConfig)
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
        raw = yaml.safe_load(f) or {}

    field_raw = raw.get("field", {})
    headland_raw = raw.get("headland", {})
    risk_raw = raw.get("risk_field", {})
    planner_raw = raw.get("planner", {})
    selection_raw = raw.get("selection", {})
    informed_raw = raw.get("informed_sampling", {})
    batch_raw = raw.get("batch", {})
    output_raw = raw.get("output", {})

    gaussians = [_parse_gaussian(g) for g in risk_raw.get("gaussians", [])]
    cell_raw = field_raw.get("cell_size_m")
    cell_size = float(cell_raw) if cell_raw is not None else None

    return AppConfig(
        seed=int(raw.get("seed", 42)),
        field=FieldConfig(
            wkt_path=str(field_raw.get("wkt_path", "wkt/ee_field_10.wkt")),
            cell_size_m=cell_size,
            auto_cell_size=bool(field_raw.get("auto_cell_size", cell_size is None)),
        ),
        headland=HeadlandConfig(width_m=float(headland_raw.get("width_m", 18.0))),
        risk_field=RiskFieldConfig(
            headland_base=float(risk_raw.get("headland_base", 1.0)),
            inner_base=float(risk_raw.get("inner_base", 0.08)),
            headland_decay_m=float(risk_raw.get("headland_decay_m", 6.0)),
            gaussians=gaussians,
            auto_hotspots=bool(risk_raw.get("auto_hotspots", not gaussians)),
            auto_hotspot_count=int(risk_raw.get("auto_hotspot_count", 2)),
            use_pass_count=bool(risk_raw.get("use_pass_count", True)),
            pass_count_weight=float(risk_raw.get("pass_count_weight", 0.25)),
            pass_count_angles=[float(a) for a in risk_raw.get("pass_count_angles", [0, 45, 90, 135])],
            repeat_penalty=float(risk_raw.get("repeat_penalty", 0.35)),
            turning_penalty=float(risk_raw.get("turning_penalty", 0.25)),
            turning_angle_deg=float(risk_raw.get("turning_angle_deg", 45.0)),
            normalize=bool(risk_raw.get("normalize", True)),
        ),
        planner=PlannerConfig(
            swath_width_m=float(planner_raw.get("swath_width_m", 6.0)),
            angle_step_deg=float(planner_raw.get("angle_step_deg", 5.0)),
            min_coverage=float(planner_raw.get("min_coverage", 0.90)),
            waypoint_spacing_m=float(planner_raw.get("waypoint_spacing_m", 2.0)),
        ),
        selection=SelectionConfig(
            delta=float(selection_raw.get("delta", 0.38)),
            lambda_weighted=float(selection_raw.get("lambda_weighted", 0.8)),
            beta_rb_ccp=float(selection_raw.get("beta_rb_ccp", 1.0)),
            beta_nr_ccp=float(selection_raw.get("beta_nr_ccp", selection_raw.get("beta_rb_ccp", 1.0))),
        ),
        informed_sampling=InformedSamplingConfig(
            enabled=bool(informed_raw.get("enabled", True)),
            coarse_step_deg=float(informed_raw.get("coarse_step_deg", 15.0)),
            fine_step_deg=float(informed_raw.get("fine_step_deg", 2.0)),
            top_k_coarse=int(informed_raw.get("top_k_coarse", 6)),
            include_principal_axes=bool(informed_raw.get("include_principal_axes", True)),
            learned_weight=float(informed_raw.get("learned_weight", 0.35)),
            model_path=informed_raw.get("model_path"),
        ),
        methods=list(
            raw.get(
                "methods",
                ["naive", "weighted", "rb_ccp", "nr_ccp", "fields2cover"],
            )
        ),
        batch=BatchConfig(
            field_glob=str(batch_raw.get("field_glob", "wkt/*.wkt")),
            max_fields=int(batch_raw.get("max_fields", 350)),
            save_figures=bool(batch_raw.get("save_figures", False)),
            progress_every=int(batch_raw.get("progress_every", 10)),
        ),
        output=OutputConfig(
            figures_dir=Path(output_raw.get("figures_dir", "outputs/figures")),
            results_dir=Path(output_raw.get("results_dir", "outputs/results")),
            advisor_demo_dir=Path(
                output_raw.get("advisor_demo_dir", "outputs/advisor_demo")
            ),
            dpi=int(output_raw.get("dpi", 150)),
        ),
    )
