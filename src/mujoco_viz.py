"""MuJoCo 3D visualization for NR-CCP coverage-path replay."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.geometry import FieldGrid
from src.simulation import SimulationResult, VehicleState
from src.trajectory import VehicleMotionConfig

MUJOCO_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "mujoco"


@dataclass(frozen=True)
class MujocoReplayConfig:
    """Viewer and playback options."""

    speed: float = 1.0
    loop: bool = True
    trail_length: int = 120
    chassis_height_m: float = 0.55
    boundary_radius_m: float = 0.35
    inner_boundary_radius_m: float = 0.22


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _local_xy(x: float, y: float, origin_x: float, origin_y: float) -> tuple[float, float]:
    return x - origin_x, y - origin_y


def _field_center_local(grid: FieldGrid) -> tuple[float, float, float]:
    minx, miny, maxx, maxy = grid.geometry.bounds
    cx = (minx + maxx) * 0.5 - grid.origin_x
    cy = (miny + maxy) * 0.5 - grid.origin_y
    return cx, cy, 0.0


def _field_span_m(grid: FieldGrid) -> float:
    minx, miny, maxx, maxy = grid.geometry.bounds
    return max(maxx - minx, maxy - miny, 20.0)


def _scene_offset(grid: FieldGrid) -> tuple[float, float]:
    """Shift the scene so the field centroid sits near MuJoCo world origin."""
    cx, cy, _ = _field_center_local(grid)
    return cx, cy


def _polygon_capsule_elements(
    polygon,
    origin_x: float,
    origin_y: float,
    scene_offset_x: float,
    scene_offset_y: float,
    name_prefix: str,
    rgba: str,
    radius: float,
    z: float,
) -> list[str]:
    coords = list(polygon.exterior.coords)
    if len(coords) < 2:
        return []

    lines: list[str] = []
    for i in range(len(coords) - 1):
        x0, y0 = _local_xy(coords[i][0], coords[i][1], origin_x, origin_y)
        x1, y1 = _local_xy(coords[i + 1][0], coords[i + 1][1], origin_x, origin_y)
        x0 -= scene_offset_x
        y0 -= scene_offset_y
        x1 -= scene_offset_x
        y1 -= scene_offset_y
        lines.append(
            f'    <geom name="{name_prefix}_{i}" type="capsule" '
            f'fromto="{x0:.4f} {y0:.4f} {z:.4f} {x1:.4f} {y1:.4f} {z:.4f}" '
            f'size="{radius:.4f}" rgba="{rgba}" contype="0" conaffinity="0"/>'
        )
    return lines


def build_scene_xml(
    grid: FieldGrid,
    motion: VehicleMotionConfig,
    replay: MujocoReplayConfig | None = None,
    assets_dir: Path | None = None,
) -> str:
    """Compose a field-specific MJCF string from the base scene + polygon outlines."""
    replay = replay or MujocoReplayConfig()
    assets_dir = assets_dir or MUJOCO_ASSETS_DIR

    base_path = assets_dir / "base_scene.xml"
    tractor_path = assets_dir / "tractor.xml"
    if not base_path.exists() or not tractor_path.exists():
        raise FileNotFoundError(f"Missing MuJoCo assets under {assets_dir}")

    base_text = base_path.read_text(encoding="utf-8")
    tractor_text = tractor_path.read_text(encoding="utf-8")

    # Inline tractor include so from_xml_string works from any cwd.
    base_text = base_text.replace('<include file="tractor.xml"/>', "")
    tractor_body = ET.fromstring(tractor_text)
    tractor_world = tractor_body.find("worldbody")
    if tractor_world is None:
        raise ValueError("tractor.xml must contain <worldbody>")

    origin_x = grid.origin_x
    origin_y = grid.origin_y
    offset_x, offset_y = _scene_offset(grid)
    span = _field_span_m(grid)

    outer_lines = _polygon_capsule_elements(
        grid.geometry.outer,
        origin_x,
        origin_y,
        offset_x,
        offset_y,
        "field_outer",
        "0.85 0.15 0.10 1",
        replay.boundary_radius_m,
        0.20,
    )
    inner_lines = _polygon_capsule_elements(
        grid.geometry.inner,
        origin_x,
        origin_y,
        offset_x,
        offset_y,
        "field_inner",
        "0.95 0.95 0.20 1",
        replay.inner_boundary_radius_m,
        0.15,
    )

    # Centroid marker helps confirm the camera target when debugging the scene.
    marker = (
        f'    <site name="field_center" type="sphere" size="{max(span * 0.015, 0.8):.3f}" '
        f'pos="0 0 0.25" rgba="1 0.2 0.2 0.85"/>'
    )

    tractor_xml = "\n".join(
        ET.tostring(child, encoding="unicode") for child in tractor_world
    )
    field_xml = "\n".join(outer_lines + inner_lines)

    insert = f"\n{tractor_xml}\n{field_xml}\n{marker}\n"
    merged = base_text.replace("  </worldbody>", f"{insert}  </worldbody>")
    return merged


def load_scene_model(
    grid: FieldGrid,
    motion: VehicleMotionConfig,
    replay: MujocoReplayConfig | None = None,
    assets_dir: Path | None = None):
    """Build and compile the MuJoCo model for a specific field."""
    import mujoco

    xml = build_scene_xml(grid, motion, replay=replay, assets_dir=assets_dir)
    return mujoco.MjModel.from_xml_string(xml)


def _set_camera(viewer, grid: FieldGrid, distance_scale: float = 1.25) -> None:
    import mujoco

    span = _field_span_m(grid)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
    viewer.cam.distance = span * distance_scale
    viewer.cam.elevation = -40
    viewer.cam.azimuth = 145


def _apply_state_to_mocap(
    data,
    mocap_id: int,
    state: VehicleState,
    origin_x: float,
    origin_y: float,
    scene_offset_x: float,
    scene_offset_y: float,
    height_m: float,
) -> None:
    lx, ly = _local_xy(state.x, state.y, origin_x, origin_y)
    data.mocap_pos[mocap_id] = [lx - scene_offset_x, ly - scene_offset_y, height_m]
    data.mocap_quat[mocap_id] = _yaw_to_quat_wxyz(state.yaw)


def _risk_rgba(risk: float) -> tuple[float, float, float, float]:
    r = float(np.clip(risk, 0.0, 1.0))
    if r < 0.5:
        t = r / 0.5
        return 0.2 + 0.6 * t, 0.85 - 0.35 * t, 0.25, 1.0
    t = (r - 0.5) / 0.5
    return 0.8 + 0.2 * t, 0.5 - 0.45 * t, 0.1, 1.0


def replay_in_viewer(
    grid: FieldGrid,
    sim: SimulationResult,
    motion: VehicleMotionConfig,
    replay: MujocoReplayConfig | None = None,
    assets_dir: Path | None = None,
    field_name: str = "field",
    method: str = "rb_ccp",
) -> None:
    """Open an interactive MuJoCo viewer and replay simulation states."""
    import mujoco
    import mujoco.viewer

    replay = replay or MujocoReplayConfig()
    states = sim.states
    if not states:
        raise ValueError("Simulation produced no states to replay")

    model = load_scene_model(grid, motion, replay=replay, assets_dir=assets_dir)
    data = mujoco.MjData(model)

    tractor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tractor")
    if tractor_body_id < 0:
        raise ValueError("Scene model is missing mocap body 'tractor'")
    mocap_id = model.body_mocapid[tractor_body_id]
    if mocap_id < 0:
        raise ValueError("Body 'tractor' must have mocap='true'")

    chassis_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
    origin_x = grid.origin_x
    origin_y = grid.origin_y
    offset_x, offset_y = _scene_offset(grid)
    span = _field_span_m(grid)
    frame_dt = 0.05 / max(replay.speed, 1e-6)

    print("\n=== MuJoCo Viewer ===")
    print(f"  field  : {field_name}")
    print(f"  method : {method}")
    print(f"  frames : {len(states)}  (~{sim.duration_s:.1f}s)")
    print(f"  span   : {span:.0f} m")
    print("  you should see:")
    print("    - green checker ground")
    print("    - red field boundary loop")
    print("    - yellow inner workable loop")
    print("    - red sphere at field center")
    print("    - blue tractor moving inside the loops")
    print("  controls:")
    print("    - drag mouse : rotate / pan / zoom")
    print("    - double-click : recenter on clicked point")
    print("    - close window to exit")

    idx = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        _set_camera(viewer, grid)
        while viewer.is_running():
            state = states[idx]
            _apply_state_to_mocap(
                data,
                mocap_id,
                state,
                origin_x,
                origin_y,
                offset_x,
                offset_y,
                replay.chassis_height_m,
            )
            if chassis_geom_id >= 0:
                model.geom_rgba[chassis_geom_id] = _risk_rgba(state.risk)

            mujoco.mj_forward(model, data)
            if idx == 0:
                _set_camera(viewer, grid)
            viewer.sync()
            time.sleep(frame_dt)

            idx += 1
            if idx >= len(states):
                if replay.loop:
                    idx = 0
                else:
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
                    break


def check_scene_loads(
    grid: FieldGrid,
    motion: VehicleMotionConfig,
    assets_dir: Path | None = None,
) -> dict:
    """Headless sanity check: compile MJCF without opening a viewer."""
    model = load_scene_model(grid, motion, assets_dir=assets_dir)
    return {
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "field_center_local": (0.0, 0.0, 0.0),
        "field_span_m": _field_span_m(grid),
    }
