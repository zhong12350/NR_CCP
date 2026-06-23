"""Evaluation metrics and CSV export helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.assessor import FeasibilityCertificate, PathAssessment
from src.selectors import SelectionResult


@dataclass
class MethodMetrics:
    field_name: str
    method: str
    angle_deg: float
    path_length_m: float
    compaction_cost: float
    mean_risk: float
    max_risk: float
    coverage_rate: float
    num_turns: int
    num_candidates: int
    num_informed_candidates: int
    candidate_pool: str
    fallback: bool
    violation: float
    delta: float
    headland_cost: float = 0.0
    hotspot_cost: float = 0.0
    pass_count_cost: float = 0.0
    repeat_cost: float = 0.0
    turn_cost: float = 0.0
    runtime_full_assess_s: float = 0.0
    runtime_nr_pool_s: float = 0.0
    field_area_m2: float = 0.0
    aspect_ratio: float = 0.0
    boundary_complexity: float = 0.0
    cert_min_violation: float = 0.0
    cert_best_coverage: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def metrics_from_selection(
    field_name: str,
    selection: SelectionResult,
    num_candidates: int,
    num_informed_candidates: int,
    delta: float,
    certificate: FeasibilityCertificate | None = None,
    runtime_full_assess_s: float = 0.0,
    runtime_nr_pool_s: float = 0.0,
) -> MethodMetrics:
    a: PathAssessment = selection.assessment
    if selection.method in ("rb_ccp", "nr_ccp"):
        fallback = selection.fallback
        violation = selection.violation
    else:
        fallback = a.mean_risk > delta + 1e-9
        violation = max(0.0, a.mean_risk - delta)

    cert = certificate
    return MethodMetrics(
        field_name=field_name,
        method=selection.method,
        angle_deg=a.angle_deg,
        path_length_m=a.path_length_m,
        compaction_cost=a.compaction_cost,
        mean_risk=a.mean_risk,
        max_risk=a.max_risk,
        coverage_rate=a.coverage_rate,
        num_turns=a.num_turns,
        num_candidates=num_candidates,
        num_informed_candidates=num_informed_candidates,
        candidate_pool=selection.candidate_pool,
        fallback=fallback,
        violation=violation,
        delta=delta,
        headland_cost=a.headland_cost,
        hotspot_cost=a.hotspot_cost,
        pass_count_cost=a.pass_count_cost,
        repeat_cost=a.repeat_cost,
        turn_cost=a.turn_cost,
        runtime_full_assess_s=runtime_full_assess_s,
        runtime_nr_pool_s=runtime_nr_pool_s,
        field_area_m2=cert.field_area_m2 if cert else 0.0,
        aspect_ratio=cert.aspect_ratio if cert else 0.0,
        boundary_complexity=cert.boundary_complexity if cert else 0.0,
        cert_min_violation=cert.min_violation if cert else 0.0,
        cert_best_coverage=cert.best_coverage if cert else 0.0,
    )
