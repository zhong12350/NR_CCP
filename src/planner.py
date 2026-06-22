"""RB-CCP / NR-CCP planning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.assessor import PathAssessment, assess_all_candidates
from src.config_loader import AppConfig
from src.fields import RiskField, build_risk_field
from src.geometry import FieldGrid, load_field_from_wkt
from src.risk_search import build_candidate_pools
from src.selectors import SelectionResult, select_all


@dataclass(frozen=True)
class FieldPlanResult:
    field_name: str
    grid: FieldGrid
    risk: RiskField
    full_candidates: list
    informed_candidates: list
    full_assessments: list[PathAssessment]
    informed_assessments: list[PathAssessment]
    selections: list[SelectionResult]

    @property
    def assessments(self) -> list[PathAssessment]:
        return self.full_assessments

    @property
    def candidates(self) -> list:
        return self.full_candidates


def plan_field(
    wkt_path: str | Path,
    config: AppConfig,
) -> FieldPlanResult:
    """
    Full pipeline:
    headland → risk field (+ pass-count) → full enum + informed search → assess → select.
    """
    path = Path(wkt_path)
    grid = load_field_from_wkt(
        path,
        config.headland.width_m,
        cell_size_m=config.field.cell_size_m,
        auto_cell_size=config.field.auto_cell_size,
    )
    risk = build_risk_field(grid, config.risk_field, config.planner)

    full_cands, informed_cands = build_candidate_pools(grid, risk, config)
    assess_kwargs = dict(
        grid=grid,
        risk=risk,
        risk_cfg=config.risk_field,
        planner_cfg=config.planner,
        lambda_weighted=config.selection.lambda_weighted,
        beta_rb=config.selection.beta_rb_ccp,
        min_coverage=config.planner.min_coverage,
    )
    full_assess = assess_all_candidates(full_cands, **assess_kwargs)
    informed_assess = assess_all_candidates(informed_cands, **assess_kwargs)

    selections = select_all(
        config.methods,
        full_assess,
        informed_assess,
        config.selection.delta,
        config.selection.lambda_weighted,
        config.selection.beta_rb_ccp,
        config.selection.beta_nr_ccp,
    )

    return FieldPlanResult(
        field_name=path.stem,
        grid=grid,
        risk=risk,
        full_candidates=full_cands,
        informed_candidates=informed_cands,
        full_assessments=full_assess,
        informed_assessments=informed_assess,
        selections=selections,
    )
