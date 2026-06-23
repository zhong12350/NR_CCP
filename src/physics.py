"""Physics-informed compaction factors: wheel load, tire pressure, soil moisture."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.config_loader import PhysicsConfig, SoilConfig, VehicleConfig


@dataclass(frozen=True)
class PhysicsFactors:
    """Per-field physics multipliers applied by the risk assessor."""

    wheel_load_n: float
    contact_pressure_kpa: float
    load_factor: float
    pressure_factor: float
    moisture_factor: float
    combined_factor: float


def compute_wheel_load_n(vehicle: VehicleConfig) -> float:
    """Vertical wheel load from vehicle mass and axle distribution."""
    return (
        vehicle.mass_kg
        * vehicle.gravity
        * vehicle.axle_load_ratio
        / max(vehicle.wheels_per_axle, 1)
    )


def compute_contact_pressure_kpa(vehicle: VehicleConfig, wheel_load_n: float) -> float:
    """
    Tire-ground contact pressure.

    Prefer geometric contact patch; fall back to inflation pressure if area is invalid.
    """
    area = vehicle.tire_width_m * vehicle.contact_length_m
    if area > 1e-6:
        return wheel_load_n / area / 1000.0
    return vehicle.tire_inflation_pressure_kpa


def compute_load_factor(wheel_load_n: float, vehicle: VehicleConfig) -> float:
    ref = max(vehicle.load_ref_n, 1.0)
    return (wheel_load_n / ref) ** vehicle.load_exponent


def compute_pressure_factor(contact_pressure_kpa: float, vehicle: VehicleConfig) -> float:
    ref = max(vehicle.pressure_ref_kpa, 1.0)
    return (contact_pressure_kpa / ref) ** vehicle.pressure_exponent


def compute_moisture_factor(soil: SoilConfig) -> float:
    """Sigmoid moisture amplification around a critical volumetric water content."""
    x = soil.moisture_steepness * (soil.moisture - soil.moisture_crit)
    sigmoid = 1.0 / (1.0 + math.exp(-x))
    return 1.0 + soil.moisture_gain * sigmoid


def compute_physics_factors(
    vehicle: VehicleConfig,
    soil: SoilConfig,
    physics: PhysicsConfig,
) -> PhysicsFactors:
    """Combine wheel load, contact pressure, and soil moisture into one multiplier."""
    if not physics.enabled:
        return PhysicsFactors(0.0, 0.0, 1.0, 1.0, 1.0, 1.0)

    wheel_load_n = compute_wheel_load_n(vehicle)
    contact_pressure_kpa = compute_contact_pressure_kpa(vehicle, wheel_load_n)
    load_factor = compute_load_factor(wheel_load_n, vehicle)
    pressure_factor = compute_pressure_factor(contact_pressure_kpa, vehicle)
    moisture_factor = compute_moisture_factor(soil)

    combined = load_factor * pressure_factor * moisture_factor
    if physics.clip_max_factor > 0:
        combined = min(combined, physics.clip_max_factor)

    return PhysicsFactors(
        wheel_load_n=wheel_load_n,
        contact_pressure_kpa=contact_pressure_kpa,
        load_factor=load_factor,
        pressure_factor=pressure_factor,
        moisture_factor=moisture_factor,
        combined_factor=combined,
    )
