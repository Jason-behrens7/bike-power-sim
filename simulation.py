"""Bike power simulation physics engine.

Calculates cycling speed from power output, rider/bike parameters,
and environmental conditions using iterative numerical methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAVITY = 9.80665  # m/s²
SEA_LEVEL_PRESSURE = 101325.0  # Pa
SEA_LEVEL_TEMP = 288.15  # K (15 °C)
LAPSE_RATE = 0.0065  # K/m
GAS_CONSTANT = 8.31447  # J/(mol·K)
MOLAR_MASS_AIR = 0.0289644  # kg/mol
SEA_LEVEL_AIR_DENSITY = 1.225  # kg/m³


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------
class TireType(Enum):
    ROAD_RACING = "Road Racing"
    ROAD_TRAINING = "Road Training"
    GRAVEL = "Gravel"
    MOUNTAIN = "Mountain"
    FAT_BIKE = "Fat Bike"


class RidingPosition(Enum):
    DROPS = "Drops"
    HOODS = "Hoods"
    TOPS = "Tops"
    AEROBARS = "Aerobars"
    MTB = "Mountain Bike"
    UPRIGHT = "Upright / City"


# Crr values by tire type
TIRE_CRR: dict[TireType, float] = {
    TireType.ROAD_RACING: 0.0030,
    TireType.ROAD_TRAINING: 0.0040,
    TireType.GRAVEL: 0.0060,
    TireType.MOUNTAIN: 0.0080,
    TireType.FAT_BIKE: 0.0120,
}

# Base CdA multipliers by position (applied to frontal-area estimate)
POSITION_CD: dict[RidingPosition, float] = {
    RidingPosition.DROPS: 0.74,
    RidingPosition.HOODS: 0.82,
    RidingPosition.TOPS: 0.90,
    RidingPosition.AEROBARS: 0.62,
    RidingPosition.MTB: 0.95,
    RidingPosition.UPRIGHT: 1.05,
}


class ForceBreakdown(NamedTuple):
    """Individual resistive force components in Newtons."""
    aero: float
    rolling: float
    gravity: float


@dataclass
class RiderParams:
    """Rider-specific parameters."""
    power_watts: float = 200.0
    weight_kg: float = 75.0
    height_cm: float = 178.0


@dataclass
class BikeParams:
    """Bike-specific parameters."""
    weight_kg: float = 8.0
    tire_type: TireType = TireType.ROAD_TRAINING
    position: RidingPosition = RidingPosition.HOODS
    drivetrain_efficiency: float = 0.97


@dataclass
class CourseParams:
    """Course / environmental parameters."""
    grade_pct: float = 0.0
    headwind_kmh: float = 0.0
    elevation_m: float = 0.0
    temperature_c: float = 20.0


@dataclass
class SimulationResult:
    """Complete result of a single simulation run."""
    speed_ms: float
    speed_kmh: float
    speed_mph: float
    forces: ForceBreakdown
    power_aero: float
    power_rolling: float
    power_gravity: float
    power_drivetrain_loss: float
    air_density: float
    cda: float
    crr: float


@dataclass
class Preset:
    """A named combination of parameters."""
    name: str
    rider: RiderParams
    bike: BikeParams
    course: CourseParams


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def air_density(elevation_m: float, temperature_c: float) -> float:
    """Calculate air density using barometric formula with temperature correction.

    Uses the international standard atmosphere model adjusted for actual
    temperature vs standard temperature at the given elevation.
    """
    temp_k = temperature_c + 273.15
    std_temp_at_elev = SEA_LEVEL_TEMP - LAPSE_RATE * elevation_m

    pressure = SEA_LEVEL_PRESSURE * (
        (1 - LAPSE_RATE * elevation_m / SEA_LEVEL_TEMP)
        ** (GRAVITY * MOLAR_MASS_AIR / (GAS_CONSTANT * LAPSE_RATE))
    )
    rho = pressure * MOLAR_MASS_AIR / (GAS_CONSTANT * temp_k)
    return rho


def estimate_frontal_area(height_cm: float) -> float:
    """Estimate rider frontal area (m²) from height.

    Empirical formula based on Heil (2001) research.
    """
    height_m = height_cm / 100.0
    return 0.0293 * height_m**0.725 * 74.0**0.425 + 0.0604


def calculate_cda(height_cm: float, position: RidingPosition) -> float:
    """Calculate drag area CdA (m²) from rider height and position."""
    frontal_area = estimate_frontal_area(height_cm)
    cd = POSITION_CD[position]
    return cd * frontal_area


def grade_to_radians(grade_pct: float) -> float:
    """Convert grade percentage to radians."""
    return math.atan(grade_pct / 100.0)


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def resistive_forces(
    speed_ms: float,
    total_mass_kg: float,
    cda: float,
    crr: float,
    rho: float,
    grade_rad: float,
    headwind_ms: float,
) -> ForceBreakdown:
    """Calculate the three main resistive forces at a given speed."""
    apparent_wind = speed_ms + headwind_ms
    f_aero = 0.5 * rho * cda * apparent_wind * abs(apparent_wind)
    f_roll = crr * total_mass_kg * GRAVITY * math.cos(grade_rad)
    f_grav = total_mass_kg * GRAVITY * math.sin(grade_rad)
    return ForceBreakdown(aero=f_aero, rolling=f_roll, gravity=f_grav)


def power_required(
    speed_ms: float,
    total_mass_kg: float,
    cda: float,
    crr: float,
    rho: float,
    grade_rad: float,
    headwind_ms: float,
    efficiency: float,
) -> float:
    """Total power required to maintain a given speed."""
    forces = resistive_forces(
        speed_ms, total_mass_kg, cda, crr, rho, grade_rad, headwind_ms
    )
    total_force = forces.aero + forces.rolling + forces.gravity
    if speed_ms <= 0:
        return 0.0
    return total_force * speed_ms / efficiency


def solve_speed(
    rider: RiderParams,
    bike: BikeParams,
    course: CourseParams,
    power_override: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> SimulationResult:
    """Solve for steady-state speed given power and conditions.

    Uses Newton-Raphson iteration to find the speed where power input
    equals power required to overcome all resistive forces.
    """
    pw = power_override if power_override is not None else rider.power_watts
    total_mass = rider.weight_kg + bike.weight_kg
    cda = calculate_cda(rider.height_cm, bike.position)
    crr = TIRE_CRR[bike.tire_type]
    rho = air_density(course.elevation_m, course.temperature_c)
    grade_rad = grade_to_radians(course.grade_pct)
    headwind_ms = course.headwind_kmh / 3.6
    eta = bike.drivetrain_efficiency

    # For zero/negative power on a descent, still solve for coasting speed
    if pw <= 0 and course.grade_pct >= 0:
        return SimulationResult(
            speed_ms=0.0,
            speed_kmh=0.0,
            speed_mph=0.0,
            forces=ForceBreakdown(0.0, 0.0, 0.0),
            power_aero=0.0,
            power_rolling=0.0,
            power_gravity=0.0,
            power_drivetrain_loss=0.0,
            air_density=rho,
            cda=cda,
            crr=crr,
        )

    # Initial guess
    if grade_rad < 0:
        # On descents, estimate equilibrium speed where aero drag balances gravity
        net_downhill_force = (
            abs(total_mass * GRAVITY * math.sin(grade_rad))
            - crr * total_mass * GRAVITY * math.cos(grade_rad)
        )
        if net_downhill_force > 0:
            # v where 0.5*rho*CdA*v^2 = net_downhill_force
            v = math.sqrt(net_downhill_force / (0.5 * rho * cda))
        else:
            v = max(1.0, (max(pw, 1) / (0.5 * rho * cda + crr * total_mass * GRAVITY)) ** (1 / 3))
    else:
        v = max(1.0, (max(pw, 1) / (0.5 * rho * cda + crr * total_mass * GRAVITY)) ** (1 / 3))

    for _ in range(max_iter):
        p_req = power_required(
            v, total_mass, cda, crr, rho, grade_rad, headwind_ms, eta
        )
        residual = p_req - pw

        # Numerical derivative dp/dv
        dv = max(0.001, v * 1e-6)
        p_req_plus = power_required(
            v + dv, total_mass, cda, crr, rho, grade_rad, headwind_ms, eta
        )
        dp_dv = (p_req_plus - p_req) / dv

        if abs(dp_dv) < 1e-12:
            break

        v_new = v - residual / dp_dv
        v_new = max(0.01, min(v_new, 100.0))  # clamp to [0.01, 100] m/s

        if abs(v_new - v) < tol:
            v = v_new
            break
        v = v_new

    # Final force breakdown
    forces = resistive_forces(
        v, total_mass, cda, crr, rho, grade_rad, headwind_ms
    )
    p_aero = forces.aero * v
    p_roll = forces.rolling * v
    p_grav = forces.gravity * v
    p_loss = pw * (1 - eta)

    return SimulationResult(
        speed_ms=v,
        speed_kmh=v * 3.6,
        speed_mph=v * 2.23694,
        forces=forces,
        power_aero=p_aero,
        power_rolling=p_roll,
        power_gravity=p_grav,
        power_drivetrain_loss=p_loss,
        air_density=rho,
        cda=cda,
        crr=crr,
    )


def speed_vs_power_curve(
    rider: RiderParams,
    bike: BikeParams,
    course: CourseParams,
    power_range: tuple[float, float] = (50, 500),
    num_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate arrays of power values and corresponding speeds.

    Returns (powers, speeds_kmh) as numpy arrays.
    """
    powers = np.linspace(power_range[0], power_range[1], num_points)
    speeds = np.array([
        solve_speed(rider, bike, course, power_override=p).speed_kmh
        for p in powers
    ])
    return powers, speeds


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: list[Preset] = [
    Preset(
        name="Flat Road Cruise",
        rider=RiderParams(power_watts=150, weight_kg=75, height_cm=178),
        bike=BikeParams(weight_kg=9, tire_type=TireType.ROAD_TRAINING,
                        position=RidingPosition.HOODS),
        course=CourseParams(grade_pct=0, headwind_kmh=0, elevation_m=100,
                            temperature_c=20),
    ),
    Preset(
        name="Hill Climb (8%)",
        rider=RiderParams(power_watts=280, weight_kg=68, height_cm=175),
        bike=BikeParams(weight_kg=7, tire_type=TireType.ROAD_RACING,
                        position=RidingPosition.HOODS),
        course=CourseParams(grade_pct=8, headwind_kmh=0, elevation_m=800,
                            temperature_c=15),
    ),
    Preset(
        name="Time Trial",
        rider=RiderParams(power_watts=320, weight_kg=78, height_cm=182),
        bike=BikeParams(weight_kg=8, tire_type=TireType.ROAD_RACING,
                        position=RidingPosition.AEROBARS),
        course=CourseParams(grade_pct=0, headwind_kmh=0, elevation_m=50,
                            temperature_c=22),
    ),
    Preset(
        name="Headwind Battle",
        rider=RiderParams(power_watts=220, weight_kg=80, height_cm=180),
        bike=BikeParams(weight_kg=9, tire_type=TireType.ROAD_TRAINING,
                        position=RidingPosition.DROPS),
        course=CourseParams(grade_pct=0, headwind_kmh=25, elevation_m=50,
                            temperature_c=18),
    ),
    Preset(
        name="Mountain Bike Trail",
        rider=RiderParams(power_watts=200, weight_kg=78, height_cm=178),
        bike=BikeParams(weight_kg=13, tire_type=TireType.MOUNTAIN,
                        position=RidingPosition.MTB),
        course=CourseParams(grade_pct=3, headwind_kmh=0, elevation_m=500,
                            temperature_c=18),
    ),
    Preset(
        name="Descent (-5%)",
        rider=RiderParams(power_watts=50, weight_kg=75, height_cm=178),
        bike=BikeParams(weight_kg=8, tire_type=TireType.ROAD_RACING,
                        position=RidingPosition.DROPS),
        course=CourseParams(grade_pct=-5, headwind_kmh=0, elevation_m=300,
                            temperature_c=20),
    ),
    Preset(
        name="Hot Day at Altitude",
        rider=RiderParams(power_watts=250, weight_kg=72, height_cm=176),
        bike=BikeParams(weight_kg=7.5, tire_type=TireType.ROAD_RACING,
                        position=RidingPosition.HOODS),
        course=CourseParams(grade_pct=2, headwind_kmh=5, elevation_m=2000,
                            temperature_c=35),
    ),
]
