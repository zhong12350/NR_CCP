"""Smoke tests for core method fixes."""

from __future__ import annotations

import sys
from pathlib import Path

from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.assessor import PathAssessment, compute_coverage_rate, filter_by_coverage, path_length
from src.candidates import CandidatePath, _generate_swath_lines
from src.config_loader import PlannerConfig, RiskFieldConfig, load_config
from src.fields import build_risk_field
from src.geometry import build_field_geometry, rasterize_geometry
from src.informed_sampling import FEATURE_DIM, InformedAngleSampler
from src.planner import plan_field
from src.selectors import select_nr_ccp


def _demo_grid():
    poly = Polygon([(0, 0), (120, 0), (120, 80), (0, 80)])
    geom = build_field_geometry(poly, headland_width_m=6.0, name="demo")
    grid = rasterize_geometry(geom, cell_size_m=2.0)
    planner = PlannerConfig(min_coverage=0.75, route_optimizer="2opt")
    risk_cfg = RiskFieldConfig(auto_hotspots=False, gaussians=[], use_pass_count=False)
    risk = build_risk_field(grid, risk_cfg, planner)
    return grid, planner, risk


def test_path_length_euclidean():
    pts = [(0.0, 0.0), (3.0, 4.0)]
    assert abs(path_length(pts) - 5.0) < 1e-9


def test_coverage_uses_polygon_sweep():
    grid, planner, _ = _demo_grid()
    pts = [(10, 10), (110, 10)]
    cov = compute_coverage_rate(pts, grid, planner.swath_width_m / 2.0)
    assert 0.0 < cov < 1.0


def test_risk_field_has_uncertainty_and_cvar_assessment():
    grid, planner, risk = _demo_grid()
    assert risk.uncertainty[grid.inner_mask].mean() > 0.0
    cand = CandidatePath(0.0, [(10, 10), (110, 10)], 1)
    from src.assessor import assess_candidate

    assessment = assess_candidate(
        cand,
        grid,
        risk,
        RiskFieldConfig(auto_hotspots=False, gaussians=[], use_pass_count=False),
        planner,
        lambda_weighted=0.8,
        beta_rb=1.0,
    )
    assert assessment.cvar_risk >= assessment.mean_risk
    assert assessment.bound_risk == assessment.cvar_risk


def test_neural_sampler_uses_full_feature_vector():
    grid, planner, risk = _demo_grid()
    sampler = InformedAngleSampler(load_config(ROOT / "configs" / "default.yaml").informed_sampling)
    x = sampler.feature_vector(grid, risk, 45.0, planner.swath_width_m)
    assert x.shape == (FEATURE_DIM,)


def test_concave_field_keeps_multiple_swath_segments():
    poly = Polygon([(0, 0), (100, 0), (100, 40), (50, 40), (50, 80), (0, 80)])
    geom = build_field_geometry(poly, headland_width_m=4.0, name="concave")
    grid = rasterize_geometry(geom, cell_size_m=2.0)
    swaths = _generate_swath_lines(grid, 0.0, 6.0)
    assert len(swaths) >= 2


def test_filter_by_coverage_no_silent_relaxation():
    cand = CandidatePath(0.0, [(0, 0), (1, 0)], 1)
    low = PathAssessment(
        candidate=cand,
        path_length_m=1.0,
        compaction_cost=1.0,
        mean_risk=0.2,
        max_risk=0.2,
        coverage_rate=0.5,
        num_turns=0,
        objective_length=1.0,
        objective_weighted=1.0,
        objective_rb=1.0,
    )
    out = filter_by_coverage([low], min_coverage=0.9)
    assert len(out) == 1
    assert out[0].coverage_rate < 0.9


def test_nr_ccp_independent_pool_smaller_than_full():
    wkt = ROOT / "wkt" / "ee_field_10.wkt"
    if not wkt.exists():
        return
    cfg = load_config(ROOT / "configs" / "default.yaml")
    result = plan_field(wkt, cfg, project_root=ROOT)
    assert len(result.informed_candidates) <= len(result.full_candidates)
    assert result.runtime_nr_pool_s >= 0.0


def test_nr_ccp_no_rb_guard():
    def mk(angle: float, risk: float, length: float) -> PathAssessment:
        c = CandidatePath(angle, [(0, 0), (length, 0)], 1)
        return PathAssessment(
            candidate=c,
            path_length_m=length,
            compaction_cost=length,
            mean_risk=risk,
            max_risk=risk,
            coverage_rate=1.0,
            num_turns=0,
            objective_length=length,
            objective_weighted=length,
            objective_rb=length,
        )

    pool = [mk(10.0, 0.30, 100.0), mk(20.0, 0.35, 80.0)]
    sel = select_nr_ccp(pool, delta=0.38, beta_nr=1.0)
    assert sel.candidate_pool == "informed"
    assert "rb_guard" not in sel.candidate_pool


if __name__ == "__main__":
    for t in [
        test_path_length_euclidean,
        test_coverage_uses_polygon_sweep,
        test_risk_field_has_uncertainty_and_cvar_assessment,
        test_neural_sampler_uses_full_feature_vector,
        test_concave_field_keeps_multiple_swath_segments,
        test_filter_by_coverage_no_silent_relaxation,
        test_nr_ccp_independent_pool_smaller_than_full,
        test_nr_ccp_no_rb_guard,
    ]:
        t()
        print(f"ok {t.__name__}")
    print("all passed")
