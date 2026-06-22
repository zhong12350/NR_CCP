"""Method selectors on shared / informed candidate pools."""

from __future__ import annotations

from dataclasses import dataclass

from src.assessor import PathAssessment


@dataclass(frozen=True)
class SelectionResult:
    method: str
    assessment: PathAssessment
    fallback: bool
    violation: float
    candidate_pool: str = "full"


def select_naive(assessments: list[PathAssessment]) -> SelectionResult:
    best = min(assessments, key=lambda a: a.path_length_m)
    return SelectionResult("naive", best, False, 0.0, "full")


def select_weighted(
    assessments: list[PathAssessment],
    lambda_weighted: float,
) -> SelectionResult:
    best = min(
        assessments,
        key=lambda a: a.path_length_m + lambda_weighted * a.compaction_cost,
    )
    return SelectionResult("weighted", best, False, 0.0, "full")


def select_fields2cover(assessments: list[PathAssessment]) -> SelectionResult:
    """Fields2Cover-like baseline: shortest coverage route on shared pool."""
    best = min(assessments, key=lambda a: a.path_length_m)
    return SelectionResult("fields2cover", best, False, 0.0, "full")


def select_rb_ccp(
    assessments: list[PathAssessment],
    delta: float,
    beta_rb: float,
) -> SelectionResult:
    feasible = [a for a in assessments if a.mean_risk <= delta + 1e-9]
    if feasible:
        best = min(feasible, key=lambda a: a.path_length_m + beta_rb * a.compaction_cost)
        return SelectionResult("rb_ccp", best, False, 0.0, "full")

    def fallback_key(a: PathAssessment) -> tuple[float, float]:
        return (max(0.0, a.mean_risk - delta), a.path_length_m + beta_rb * a.compaction_cost)

    best = min(assessments, key=fallback_key)
    return SelectionResult(
        "rb_ccp",
        best,
        True,
        max(0.0, best.mean_risk - delta),
        "full",
    )


def select_nr_ccp(
    assessments: list[PathAssessment],
    delta: float,
    beta_nr: float,
) -> SelectionResult:
    """NR-CCP: RB-CCP rule on informed-search candidate pool."""
    if not assessments:
        raise RuntimeError("NR-CCP informed pool is empty.")
    feasible = [a for a in assessments if a.mean_risk <= delta + 1e-9]
    if feasible:
        best = min(feasible, key=lambda a: a.path_length_m + beta_nr * a.compaction_cost)
        return SelectionResult("nr_ccp", best, False, 0.0, "informed")

    def fallback_key(a: PathAssessment) -> tuple[float, float, float]:
        return (
            max(0.0, a.mean_risk - delta),
            a.compaction_cost,
            a.path_length_m + beta_nr * a.compaction_cost,
        )

    best = min(assessments, key=fallback_key)
    return SelectionResult(
        "nr_ccp",
        best,
        True,
        max(0.0, best.mean_risk - delta),
        "informed",
    )


SELECTOR_REGISTRY = {
    "naive": select_naive,
    "weighted": select_weighted,
    "rb_ccp": select_rb_ccp,
    "nr_ccp": select_nr_ccp,
    "fields2cover": select_fields2cover,
}


def select_all(
    methods: list[str],
    full_assessments: list[PathAssessment],
    informed_assessments: list[PathAssessment],
    delta: float,
    lambda_weighted: float,
    beta_rb: float,
    beta_nr: float,
) -> list[SelectionResult]:
    if not full_assessments:
        raise RuntimeError("No feasible candidates in full pool.")

    results: list[SelectionResult] = []
    for method in methods:
        if method == "naive":
            results.append(select_naive(full_assessments))
        elif method == "weighted":
            results.append(select_weighted(full_assessments, lambda_weighted))
        elif method == "fields2cover":
            results.append(select_fields2cover(full_assessments))
        elif method == "rb_ccp":
            results.append(select_rb_ccp(full_assessments, delta, beta_rb))
        elif method == "nr_ccp":
            pool = informed_assessments or full_assessments
            results.append(select_nr_ccp(pool, delta, beta_nr))
        else:
            raise ValueError(f"Unknown method '{method}'. Available: {list(SELECTOR_REGISTRY)}")
    return results
