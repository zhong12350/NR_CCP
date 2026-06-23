"""Fields2Cover official / imported baseline integration."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.assessor import PathAssessment, assess_candidate
from src.candidates import CandidatePath, enumerate_candidates, generate_candidate
from src.config_loader import AppConfig
from src.fields import RiskField
from src.geometry import FieldGrid
from src.physics import PhysicsFactors


@dataclass(frozen=True)
class Fields2CoverRecord:
    field_name: str
    angle_deg: float
    path_length_m: float
    source: str


def _choose_f2c_angle(grid: FieldGrid, swath_width_m: float) -> float:
    """Heuristic F2C-like angle: minimize swath count, tie-break by swath length."""
    from src.candidates import _generate_swath_lines

    best_angle = 0.0
    best_key = (float("inf"), float("inf"))
    for angle in range(0, 180, 5):
        swaths = _generate_swath_lines(grid, float(angle), swath_width_m)
        if not swaths:
            continue
        total_len = sum(s.length for s in swaths)
        key = (len(swaths), total_len)
        if key < best_key:
            best_key = key
            best_angle = float(angle)
    return best_angle


def load_official_csv(path: Path) -> dict[str, Fields2CoverRecord]:
    if not path.exists():
        return {}
    records: dict[str, Fields2CoverRecord] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["field_name"]
            records[name] = Fields2CoverRecord(
                field_name=name,
                angle_deg=float(row.get("angle_deg", 0.0)),
                path_length_m=float(row.get("path_length_m", 0.0)),
                source=row.get("source", "official_csv"),
            )
    return records


def try_run_fields2cover_cli(wkt_path: Path, output_csv: Path) -> bool:
    """Optional external Fields2Cover CLI hook."""
    cmd = ["fields2cover", "plan", str(wkt_path), "--out", str(output_csv)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        return output_csv.exists()
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False


def resolve_fields2cover_assessment(
    field_name: str,
    grid: FieldGrid,
    risk: RiskField,
    config: AppConfig,
    official_records: dict[str, Fields2CoverRecord],
    full_assessments: list[PathAssessment],
    physics_factors: PhysicsFactors | None = None,
) -> PathAssessment:
    """
    Resolve Fields2Cover baseline:
    1) official imported CSV if available
    2) F2C-like swath-count angle heuristic on shared generator
    3) fallback to naive min-length from full pool
    """
    if field_name in official_records:
        rec = official_records[field_name]
        cand = generate_candidate(grid, rec.angle_deg, config.planner)
        if cand is None:
            cand = CandidatePath(rec.angle_deg, [], 0)
        assessed = assess_candidate(
            cand,
            grid,
            risk,
            config.risk_field,
            config.planner,
            config.selection.lambda_weighted,
            config.selection.beta_rb_ccp,
            physics_factors=physics_factors,
        )
        return assessed

    angle = _choose_f2c_angle(grid, config.planner.swath_width_m)
    cand = generate_candidate(grid, angle, config.planner)
    if cand is not None:
        return assess_candidate(
            cand,
            grid,
            risk,
            config.risk_field,
            config.planner,
            config.selection.lambda_weighted,
            config.selection.beta_rb_ccp,
            physics_factors=physics_factors,
        )

    return min(full_assessments, key=lambda a: a.path_length_m)


def load_official_records(project_root: Path) -> dict[str, Fields2CoverRecord]:
    csv_path = project_root / "data" / "fields2cover" / "official_results.csv"
    return load_official_csv(csv_path)
