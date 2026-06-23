"""RB-CCP / NR-CCP planning pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.assessor import (
    FeasibilityCertificate,
    PathAssessment,
    assess_all_candidates,
    build_feasibility_certificate,
)
from src.candidates import enumerate_candidates
from src.config_loader import AppConfig
from src.fields import RiskField, build_risk_field
from src.fields2cover_baseline import load_official_records, resolve_fields2cover_assessment
from src.geometry import FieldGrid, load_field_from_wkt
from src.physics import compute_physics_factors
from src.risk_search import build_candidate_pools
from src.selectors import SelectionResult, select_all, select_rb_ccp


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
    certificate: FeasibilityCertificate
    runtime_full_assess_s: float
    runtime_nr_pool_s: float

    @property
    def assessments(self) -> list[PathAssessment]:
        return self.full_assessments

    @property
    def candidates(self) -> list:
        return self.full_candidates


def plan_field(
    wkt_path: str | Path,
    config: AppConfig,
    project_root: Path | None = None,
    official_records: dict | None = None,
) -> FieldPlanResult:
    path = Path(wkt_path)
    grid = load_field_from_wkt(
        path,
        config.headland.width_m,
        cell_size_m=config.field.cell_size_m,
        auto_cell_size=config.field.auto_cell_size,
    )
    risk = build_risk_field(grid, config.risk_field, config.planner)
    physics_factors = compute_physics_factors(config.vehicle, config.soil, config.physics)

    full_cands = enumerate_candidates(grid, config.planner)
    assess_kwargs = dict(
        grid=grid,
        risk=risk,
        risk_cfg=config.risk_field,
        planner_cfg=config.planner,
        lambda_weighted=config.selection.lambda_weighted,
        beta_rb=config.selection.beta_rb_ccp,
        min_coverage=config.planner.min_coverage,
        physics_factors=physics_factors,
    )

    t0 = time.perf_counter()
    full_assess = assess_all_candidates(full_cands, **assess_kwargs)
    runtime_full = time.perf_counter() - t0

    rb_sel = (
        select_rb_ccp(full_assess, config.selection.delta, config.selection.beta_rb_ccp)
        if full_assess
        else None
    )

    t1 = time.perf_counter()
    _, informed_cands, informed_assess = build_candidate_pools(
        grid,
        risk,
        config,
        full_assessments=full_assess,
        rb_ccp_assessment=rb_sel.assessment if rb_sel else None,
    )
    runtime_nr = time.perf_counter() - t1

    selections = select_all(
        config.methods,
        full_assess,
        informed_assess,
        config.selection.delta,
        config.selection.lambda_weighted,
        config.selection.beta_rb_ccp,
        config.selection.beta_nr_ccp,
    )

    if "fields2cover" in config.methods and full_assess:
        root = project_root or Path.cwd()
        records = official_records if official_records is not None else load_official_records(root)
        f2c_assess = resolve_fields2cover_assessment(
            path.stem,
            grid,
            risk,
            config,
            records,
            full_assess,
            physics_factors=physics_factors,
        )
        selections = [
            s
            if s.method != "fields2cover"
            else SelectionResult(
                "fields2cover", f2c_assess, False, 0.0, "official_or_heuristic"
            )
            for s in selections
        ]

    certificate = build_feasibility_certificate(
        path.stem, grid, full_assess, informed_assess, config.selection.delta
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
        certificate=certificate,
        runtime_full_assess_s=runtime_full,
        runtime_nr_pool_s=runtime_nr,
    )
