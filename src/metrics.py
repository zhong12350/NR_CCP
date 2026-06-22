"""Evaluation metrics and CSV export helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.assessor import PathAssessment
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

    def to_dict(self) -> dict:
        return asdict(self)


def metrics_from_selection(
    field_name: str,
    selection: SelectionResult,
    num_candidates: int,
    num_informed_candidates: int,
    delta: float,
) -> MethodMetrics:
    a: PathAssessment = selection.assessment
    if selection.method == "rb_ccp":
        fallback = selection.fallback
        violation = selection.violation
    elif selection.method == "nr_ccp":
        fallback = selection.fallback
        violation = selection.violation
    else:
        fallback = a.mean_risk > delta + 1e-9
        violation = max(0.0, a.mean_risk - delta)
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
    )
