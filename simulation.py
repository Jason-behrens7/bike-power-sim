"""Bike power simulation physics engine.

Calculates cycling speed from power output, rider/bike parameters,
and environmental conditions using iterative numerical methods.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
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
DEFAULT_WHEEL_MASS = 1.8  # kg per wheel
DEFAULT_WHEEL_RADIUS = 0.34  # m (700c)
METABOLIC_EFFICIENCY = 0.25  # ~25% of metabolic energy becomes pedal power
KJ_TO_KCAL = 0.239006


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


TIRE_CRR: dict[TireType, float] = {
    TireType.ROAD_RACING: 0.0030,
    TireType.ROAD_TRAINING: 0.0040,
    TireType.GRAVEL: 0.0060,
    TireType.MOUNTAIN: 0.0080,
    TireType.FAT_BIKE: 0.0120,
}

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
    wheel_mass_kg: float = DEFAULT_WHEEL_MASS
    wheel_radius_m: float = DEFAULT_WHEEL_RADIUS


@dataclass
class CourseParams:
    """Course / environmental parameters."""
    grade_pct: float = 0.0
    headwind_kmh: float = 0.0
    wind_direction_deg: float = 0.0
    elevation_m: float = 0.0
    temperature_c: float = 20.0


@dataclass
class CourseSegment:
    """A single segment in a multi-segment course profile."""
    distance_m: float = 1000.0
    grade_pct: float = 0.0
    headwind_kmh: float = 0.0
    wind_direction_deg: float = 0.0
    elevation_m: float = 0.0
    temperature_c: float = 20.0


@dataclass
class SegmentResult:
    """Result for a single course segment."""
    segment_index: int
    distance_m: float
    grade_pct: float
    speed_kmh: float
    speed_ms: float
    time_s: float
    elevation_start_m: float
    elevation_end_m: float
    power_aero: float
    power_rolling: float
    power_gravity: float


@dataclass
class CourseProfileResult:
    """Result for a full course profile simulation."""
    segments: list[SegmentResult]
    total_distance_m: float
    total_time_s: float
    avg_speed_kmh: float
    total_elevation_gain_m: float
    total_elevation_loss_m: float
    distances_cumulative: list[float]
    speeds: list[float]
    elevations: list[float]


@dataclass
class RaceCompetitor:
    """A virtual competitor for race simulation."""
    name: str
    power_watts: float
    weight_kg: float
    height_cm: float = 178.0
    bike_weight_kg: float = 8.0
    tire_type: TireType = TireType.ROAD_TRAINING
    position: RidingPosition = RidingPosition.HOODS
    drivetrain_efficiency: float = 0.97


@dataclass
class RaceResult:
    """Result for a single competitor in a race simulation."""
    name: str
    total_time_s: float
    avg_speed_kmh: float
    segment_speeds: list[float]
    segment_times: list[float]
    cumulative_distances: list[float]
    cumulative_times: list[float]
    gap_to_leader_s: float = 0.0


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
# Interval / workout
# ---------------------------------------------------------------------------
@dataclass
class IntervalStep:
    """One step in an interval workout."""
    power_watts: float = 200.0
    duration_s: float = 300.0


@dataclass
class IntervalPoint:
    """A single time-point result from an interval simulation."""
    time_s: float
    power_watts: float
    speed_kmh: float
    distance_m: float
    calories_kcal: float


@dataclass
class WorkoutResult:
    """Full result of an interval workout simulation."""
    points: list[IntervalPoint]
    total_time_s: float
    total_distance_m: float
    total_calories_kcal: float
    total_work_kj: float
    avg_speed_kmh: float
    avg_power_watts: float


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------
@dataclass
class LeaderboardEntry:
    """A single leaderboard entry."""
    course_name: str
    rider_name: str
    time_s: float
    avg_speed_kmh: float
    power_watts: float
    date: str


# ---------------------------------------------------------------------------
# Power zones
# ---------------------------------------------------------------------------
ZONE_NAMES = ["Z1 Recovery", "Z2 Endurance", "Z3 Tempo",
              "Z4 Threshold", "Z5 VO2max", "Z6 Anaerobic", "Z7 Sprint"]
ZONE_COLORS = ["#89b4fa", "#a6e3a1", "#f9e2af", "#fab387",
               "#f38ba8", "#cba6f7", "#f5c2e7"]

# Rider classification based on W/kg (male FTP benchmarks)
RIDER_CATEGORIES = [
    ("World Tour Pro", 6.0, float("inf")),
    ("Cat 1 / Elite", 5.0, 6.0),
    ("Cat 2", 4.2, 5.0),
    ("Cat 3", 3.5, 4.2),
    ("Cat 4", 2.8, 3.5),
    ("Cat 5 / Beginner", 0.0, 2.8),
]


def power_zones(ftp: float) -> list[tuple[str, float, float, str]]:
    """Return power zones as (name, low_watts, high_watts, color) based on FTP."""
    boundaries = [0, 0.55, 0.75, 0.90, 1.05, 1.20, 1.50, 999.0]
    zones = []
    for i, name in enumerate(ZONE_NAMES):
        lo = ftp * boundaries[i]
        hi = ftp * boundaries[i + 1]
        zones.append((name, lo, hi, ZONE_COLORS[i]))
    return zones


def zone_for_power(power: float, ftp: float) -> tuple[str, str]:
    """Return (zone_name, zone_color) for a given power and FTP."""
    zones = power_zones(ftp)
    for name, lo, hi, color in zones:
        if lo <= power < hi:
            return name, color
    return ZONE_NAMES[-1], ZONE_COLORS[-1]


def power_to_weight_ratio(power_watts: float, rider_weight_kg: float) -> float:
    """Calculate power-to-weight ratio in W/kg."""
    if rider_weight_kg <= 0:
        return 0.0
    return power_watts / rider_weight_kg


def classify_rider(w_per_kg: float) -> str:
    """Classify rider based on W/kg into racing categories."""
    for name, lo, hi in RIDER_CATEGORIES:
        if lo <= w_per_kg < hi:
            return name
    return RIDER_CATEGORIES[-1][0]


def w_per_kg_analysis(ftp: float, rider_weight_kg: float) -> dict:
    """Provide comprehensive power-to-weight analysis."""
    w_kg = power_to_weight_ratio(ftp, rider_weight_kg)
    category = classify_rider(w_kg)
    return {
        "w_per_kg": w_kg,
        "category": category,
        "ftp": ftp,
        "weight_kg": rider_weight_kg,
        "categories": [(n, lo, hi) for n, lo, hi in RIDER_CATEGORIES],
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "power_watts": (0, 3000),
    "rider_weight_kg": (20, 250),
    "rider_height_cm": (100, 230),
    "bike_weight_kg": (1, 50),
    "drivetrain_efficiency": (0.70, 1.0),
    "grade_pct": (-45, 45),
    "headwind_kmh": (-100, 100),
    "wind_direction_deg": (0, 360),
    "elevation_m": (-500, 9000),
    "temperature_c": (-40, 60),
    "wheel_mass_kg": (0.2, 5.0),
    "wheel_radius_m": (0.15, 0.45),
    "segment_distance_m": (10, 500000),
}


def validate_param(name: str, value: float) -> str | None:
    """Return an error message if value is out of range, else None."""
    bounds = PARAM_RANGES.get(name)
    if bounds is None:
        return None
    lo, hi = bounds
    if not (lo <= value <= hi):
        return f"{name}: {value} is out of range [{lo}, {hi}]"
    return None


def validate_all(rider: RiderParams, bike: BikeParams,
                 course: CourseParams) -> list[str]:
    """Validate all parameters; return list of error messages."""
    errors: list[str] = []
    checks = [
        ("power_watts", rider.power_watts),
        ("rider_weight_kg", rider.weight_kg),
        ("rider_height_cm", rider.height_cm),
        ("bike_weight_kg", bike.weight_kg),
        ("drivetrain_efficiency", bike.drivetrain_efficiency),
        ("grade_pct", course.grade_pct),
        ("headwind_kmh", course.headwind_kmh),
        ("wind_direction_deg", course.wind_direction_deg),
        ("elevation_m", course.elevation_m),
        ("temperature_c", course.temperature_c),
        ("wheel_mass_kg", bike.wheel_mass_kg),
        ("wheel_radius_m", bike.wheel_radius_m),
    ]
    for name, val in checks:
        err = validate_param(name, val)
        if err:
            errors.append(err)
    return errors


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
def km_to_miles(km: float) -> float:
    return km * 0.621371

def miles_to_km(mi: float) -> float:
    return mi / 0.621371

def kg_to_lbs(kg: float) -> float:
    return kg * 2.20462

def lbs_to_kg(lbs: float) -> float:
    return lbs / 2.20462

def cm_to_inches(cm: float) -> float:
    return cm / 2.54

def inches_to_cm(inches: float) -> float:
    return inches * 2.54

def m_to_feet(m: float) -> float:
    return m * 3.28084

def feet_to_m(ft: float) -> float:
    return ft / 3.28084

def celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0

def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0

def kmh_to_mph(kmh: float) -> float:
    return kmh * 0.621371

def mph_to_kmh(mph: float) -> float:
    return mph / 0.621371


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def air_density(elevation_m: float, temperature_c: float) -> float:
    """Calculate air density using barometric formula with temperature correction."""
    temp_k = temperature_c + 273.15
    pressure = SEA_LEVEL_PRESSURE * (
        (1 - LAPSE_RATE * elevation_m / SEA_LEVEL_TEMP)
        ** (GRAVITY * MOLAR_MASS_AIR / (GAS_CONSTANT * LAPSE_RATE))
    )
    rho = pressure * MOLAR_MASS_AIR / (GAS_CONSTANT * temp_k)
    return rho


def estimate_frontal_area(height_cm: float) -> float:
    """Estimate rider frontal area (m²) from height (Heil 2001)."""
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


def effective_headwind(wind_speed_kmh: float, wind_direction_deg: float) -> float:
    """Calculate effective headwind component from wind speed and direction."""
    rad = math.radians(wind_direction_deg)
    return wind_speed_kmh * math.cos(rad)


def wheel_inertia_factor(wheel_mass_kg: float, wheel_radius_m: float,
                          total_mass_kg: float) -> float:
    """Calculate effective mass multiplier accounting for wheel rotational inertia."""
    i_wheels = 2 * wheel_mass_kg * wheel_radius_m ** 2
    return 1.0 + i_wheels / (total_mass_kg * wheel_radius_m ** 2)


def estimate_calories(power_watts: float, duration_s: float) -> float:
    """Estimate calories burned from power output and duration.

    Assumes ~25% gross metabolic efficiency (75% lost as heat).
    """
    work_kj = power_watts * duration_s / 1000.0
    metabolic_kj = work_kj / METABOLIC_EFFICIENCY
    return metabolic_kj * KJ_TO_KCAL


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
    cda_override: float | None = None,
    crr_override: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> SimulationResult:
    """Solve for steady-state speed given power and conditions.

    ``cda_override`` and ``crr_override`` allow callers (e.g. parameter
    sweeps) to pin the drag area / rolling resistance directly. When left
    as ``None`` they are derived from the rider/bike exactly as before, so
    the physics is unchanged for existing callers.
    """
    pw = power_override if power_override is not None else rider.power_watts
    total_mass = rider.weight_kg + bike.weight_kg
    inertia_factor = wheel_inertia_factor(
        bike.wheel_mass_kg, bike.wheel_radius_m, total_mass
    )
    effective_mass = total_mass * inertia_factor
    cda = cda_override if cda_override is not None else calculate_cda(rider.height_cm, bike.position)
    crr = crr_override if crr_override is not None else TIRE_CRR[bike.tire_type]
    rho = air_density(course.elevation_m, course.temperature_c)
    grade_rad = grade_to_radians(course.grade_pct)
    hw = effective_headwind(course.headwind_kmh, course.wind_direction_deg)
    headwind_ms = hw / 3.6
    eta = bike.drivetrain_efficiency

    if pw <= 0 and course.grade_pct >= 0:
        return SimulationResult(
            speed_ms=0.0, speed_kmh=0.0, speed_mph=0.0,
            forces=ForceBreakdown(0.0, 0.0, 0.0),
            power_aero=0.0, power_rolling=0.0, power_gravity=0.0,
            power_drivetrain_loss=0.0,
            air_density=rho, cda=cda, crr=crr,
        )

    if grade_rad < 0:
        net_downhill_force = (
            abs(effective_mass * GRAVITY * math.sin(grade_rad))
            - crr * effective_mass * GRAVITY * math.cos(grade_rad)
        )
        if net_downhill_force > 0:
            v = math.sqrt(net_downhill_force / (0.5 * rho * cda))
        else:
            v = max(1.0, (max(pw, 1) / (0.5 * rho * cda + crr * effective_mass * GRAVITY)) ** (1 / 3))
    else:
        v = max(1.0, (max(pw, 1) / (0.5 * rho * cda + crr * effective_mass * GRAVITY)) ** (1 / 3))

    for _ in range(max_iter):
        p_req = power_required(
            v, effective_mass, cda, crr, rho, grade_rad, headwind_ms, eta
        )
        residual = p_req - pw
        dv = max(0.001, v * 1e-6)
        p_req_plus = power_required(
            v + dv, effective_mass, cda, crr, rho, grade_rad, headwind_ms, eta
        )
        dp_dv = (p_req_plus - p_req) / dv
        if abs(dp_dv) < 1e-12:
            break
        v_new = v - residual / dp_dv
        v_new = max(0.01, min(v_new, 100.0))
        if abs(v_new - v) < tol:
            v = v_new
            break
        v = v_new

    forces = resistive_forces(
        v, effective_mass, cda, crr, rho, grade_rad, headwind_ms
    )
    p_aero = forces.aero * v
    p_roll = forces.rolling * v
    p_grav = forces.gravity * v
    p_loss = pw * (1 - eta)

    return SimulationResult(
        speed_ms=v, speed_kmh=v * 3.6, speed_mph=v * 2.23694,
        forces=forces,
        power_aero=p_aero, power_rolling=p_roll, power_gravity=p_grav,
        power_drivetrain_loss=p_loss,
        air_density=rho, cda=cda, crr=crr,
    )


def speed_vs_power_curve(
    rider: RiderParams,
    bike: BikeParams,
    course: CourseParams,
    power_range: tuple[float, float] = (50, 500),
    num_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate arrays of power values and corresponding speeds."""
    powers = np.linspace(power_range[0], power_range[1], num_points)
    speeds = np.array([
        solve_speed(rider, bike, course, power_override=p).speed_kmh
        for p in powers
    ])
    return powers, speeds


# ---------------------------------------------------------------------------
# Course profile simulation
# ---------------------------------------------------------------------------

def simulate_course_profile(
    rider: RiderParams,
    bike: BikeParams,
    segments: list[CourseSegment],
) -> CourseProfileResult:
    """Simulate a multi-segment course and return detailed results."""
    results: list[SegmentResult] = []
    cumulative_dist = 0.0
    cumulative_time = 0.0
    current_elevation = segments[0].elevation_m if segments else 0.0
    total_gain = 0.0
    total_loss = 0.0

    distances_cum: list[float] = [0.0]
    speeds_list: list[float] = []
    elevations_list: list[float] = [current_elevation]

    for i, seg in enumerate(segments):
        course = CourseParams(
            grade_pct=seg.grade_pct,
            headwind_kmh=seg.headwind_kmh,
            wind_direction_deg=seg.wind_direction_deg,
            elevation_m=seg.elevation_m,
            temperature_c=seg.temperature_c,
        )
        result = solve_speed(rider, bike, course)

        time_s = seg.distance_m / max(result.speed_ms, 0.1)
        elev_change = seg.distance_m * math.sin(grade_to_radians(seg.grade_pct))
        end_elevation = current_elevation + elev_change

        if elev_change > 0:
            total_gain += elev_change
        else:
            total_loss += abs(elev_change)

        seg_result = SegmentResult(
            segment_index=i,
            distance_m=seg.distance_m,
            grade_pct=seg.grade_pct,
            speed_kmh=result.speed_kmh,
            speed_ms=result.speed_ms,
            time_s=time_s,
            elevation_start_m=current_elevation,
            elevation_end_m=end_elevation,
            power_aero=result.power_aero,
            power_rolling=result.power_rolling,
            power_gravity=result.power_gravity,
        )
        results.append(seg_result)

        cumulative_dist += seg.distance_m
        cumulative_time += time_s
        current_elevation = end_elevation

        distances_cum.append(cumulative_dist)
        speeds_list.append(result.speed_kmh)
        elevations_list.append(end_elevation)

    avg_speed = (cumulative_dist / cumulative_time * 3.6) if cumulative_time > 0 else 0.0

    return CourseProfileResult(
        segments=results,
        total_distance_m=cumulative_dist,
        total_time_s=cumulative_time,
        avg_speed_kmh=avg_speed,
        total_elevation_gain_m=total_gain,
        total_elevation_loss_m=total_loss,
        distances_cumulative=distances_cum,
        speeds=speeds_list,
        elevations=elevations_list,
    )


# ---------------------------------------------------------------------------
# Interval / workout simulation
# ---------------------------------------------------------------------------

def simulate_workout(
    steps: list[IntervalStep],
    rider: RiderParams,
    bike: BikeParams,
    course: CourseParams,
    time_resolution_s: float = 1.0,
) -> WorkoutResult:
    """Simulate an interval workout and return time-series results."""
    points: list[IntervalPoint] = []
    cumulative_time = 0.0
    cumulative_dist = 0.0
    cumulative_cal = 0.0
    total_work_kj = 0.0
    total_power_time = 0.0

    for step in steps:
        t = 0.0
        while t < step.duration_s:
            dt = min(time_resolution_s, step.duration_s - t)
            result = solve_speed(rider, bike, course,
                                 power_override=step.power_watts)
            dist = result.speed_ms * dt
            cal = estimate_calories(step.power_watts, dt)

            cumulative_time += dt
            cumulative_dist += dist
            cumulative_cal += cal
            total_work_kj += step.power_watts * dt / 1000.0
            total_power_time += step.power_watts * dt

            points.append(IntervalPoint(
                time_s=cumulative_time,
                power_watts=step.power_watts,
                speed_kmh=result.speed_kmh,
                distance_m=cumulative_dist,
                calories_kcal=cumulative_cal,
            ))
            t += dt

    avg_speed = (cumulative_dist / cumulative_time * 3.6) if cumulative_time > 0 else 0.0
    avg_power = (total_power_time / cumulative_time) if cumulative_time > 0 else 0.0

    return WorkoutResult(
        points=points,
        total_time_s=cumulative_time,
        total_distance_m=cumulative_dist,
        total_calories_kcal=cumulative_cal,
        total_work_kj=total_work_kj,
        avg_speed_kmh=avg_speed,
        avg_power_watts=avg_power,
    )


# ---------------------------------------------------------------------------
# GPX parsing
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two lat/lon points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_gpx(filepath: str | Path) -> list[CourseSegment]:
    """Parse a GPX file and return a list of CourseSegments.

    Groups consecutive trackpoints into segments with averaged grade.
    Uses ~200m minimum segment length to avoid noise.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    trackpoints: list[tuple[float, float, float]] = []
    for trkpt in root.iter(f"{ns}trkpt"):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])
        ele_elem = trkpt.find(f"{ns}ele")
        ele = float(ele_elem.text) if ele_elem is not None else 0.0
        trackpoints.append((lat, lon, ele))

    if len(trackpoints) < 2:
        return []

    min_seg_len = 200.0
    segments: list[CourseSegment] = []
    seg_start = 0

    for i in range(1, len(trackpoints)):
        dist_so_far = 0.0
        for j in range(seg_start, i):
            dist_so_far += _haversine(
                trackpoints[j][0], trackpoints[j][1],
                trackpoints[j + 1][0], trackpoints[j + 1][1]
            )

        if dist_so_far >= min_seg_len or i == len(trackpoints) - 1:
            if dist_so_far > 0:
                elev_change = trackpoints[i][2] - trackpoints[seg_start][2]
                grade = (elev_change / dist_so_far) * 100.0
                avg_elev = (trackpoints[seg_start][2] + trackpoints[i][2]) / 2.0
                segments.append(CourseSegment(
                    distance_m=dist_so_far,
                    grade_pct=grade,
                    elevation_m=avg_elev,
                    temperature_c=20.0,
                ))
            seg_start = i

    return segments


# ---------------------------------------------------------------------------
# Race simulation
# ---------------------------------------------------------------------------

def simulate_race(
    competitors: list[RaceCompetitor],
    segments: list[CourseSegment],
) -> list[RaceResult]:
    """Simulate a race with multiple competitors on the same course."""
    results: list[RaceResult] = []

    for comp in competitors:
        rider = RiderParams(
            power_watts=comp.power_watts,
            weight_kg=comp.weight_kg,
            height_cm=comp.height_cm,
        )
        bike = BikeParams(
            weight_kg=comp.bike_weight_kg,
            tire_type=comp.tire_type,
            position=comp.position,
            drivetrain_efficiency=comp.drivetrain_efficiency,
        )

        seg_speeds: list[float] = []
        seg_times: list[float] = []
        cum_dists: list[float] = [0.0]
        cum_times: list[float] = [0.0]
        total_time = 0.0
        total_dist = 0.0

        for seg in segments:
            course = CourseParams(
                grade_pct=seg.grade_pct,
                headwind_kmh=seg.headwind_kmh,
                wind_direction_deg=seg.wind_direction_deg,
                elevation_m=seg.elevation_m,
                temperature_c=seg.temperature_c,
            )
            result = solve_speed(rider, bike, course,
                                 power_override=comp.power_watts)
            time_s = seg.distance_m / max(result.speed_ms, 0.1)
            seg_speeds.append(result.speed_kmh)
            seg_times.append(time_s)
            total_time += time_s
            total_dist += seg.distance_m
            cum_dists.append(total_dist)
            cum_times.append(total_time)

        avg_speed = (total_dist / total_time * 3.6) if total_time > 0 else 0.0

        results.append(RaceResult(
            name=comp.name,
            total_time_s=total_time,
            avg_speed_kmh=avg_speed,
            segment_speeds=seg_speeds,
            segment_times=seg_times,
            cumulative_distances=cum_dists,
            cumulative_times=cum_times,
        ))

    if results:
        best_time = min(r.total_time_s for r in results)
        for r in results:
            r.gap_to_leader_s = r.total_time_s - best_time

    results.sort(key=lambda r: r.total_time_s)
    return results


# ---------------------------------------------------------------------------
# GPX lat/lon extraction for gradient map
# ---------------------------------------------------------------------------

def parse_gpx_with_coords(filepath: str | Path) -> list[dict]:
    """Parse a GPX file and return trackpoints with lat, lon, ele, distance."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    trackpoints: list[tuple[float, float, float]] = []
    for trkpt in root.iter(f"{ns}trkpt"):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])
        ele_elem = trkpt.find(f"{ns}ele")
        ele = float(ele_elem.text) if ele_elem is not None else 0.0
        trackpoints.append((lat, lon, ele))

    if len(trackpoints) < 2:
        return []

    result: list[dict] = []
    cumulative_dist = 0.0
    result.append({
        "lat": trackpoints[0][0],
        "lon": trackpoints[0][1],
        "ele": trackpoints[0][2],
        "distance_m": 0.0,
        "grade_pct": 0.0,
    })

    for i in range(1, len(trackpoints)):
        dist = _haversine(
            trackpoints[i - 1][0], trackpoints[i - 1][1],
            trackpoints[i][0], trackpoints[i][1],
        )
        cumulative_dist += dist
        elev_change = trackpoints[i][2] - trackpoints[i - 1][2]
        grade = (elev_change / dist * 100.0) if dist > 0 else 0.0
        result.append({
            "lat": trackpoints[i][0],
            "lon": trackpoints[i][1],
            "ele": trackpoints[i][2],
            "distance_m": cumulative_dist,
            "grade_pct": grade,
        })

    return result


# ---------------------------------------------------------------------------
# Critical Power Model (CP & W')
# ---------------------------------------------------------------------------

@dataclass
class CriticalPowerModel:
    """2-parameter critical power model (CP + W')."""
    cp_watts: float
    w_prime_joules: float

    def max_power_for_duration(self, duration_s: float) -> float:
        """Max sustainable power for a given duration."""
        if duration_s <= 0:
            return self.cp_watts + self.w_prime_joules
        return self.cp_watts + self.w_prime_joules / duration_s

    def time_to_exhaustion(self, power_watts: float) -> float:
        """Time until W' is depleted at given power above CP."""
        if power_watts <= self.cp_watts:
            return float("inf")
        return self.w_prime_joules / (power_watts - self.cp_watts)

    def w_prime_balance(self, power_watts: float, duration_s: float) -> float:
        """Remaining W' after exercising at given power for duration."""
        if power_watts <= self.cp_watts:
            return self.w_prime_joules
        expenditure = (power_watts - self.cp_watts) * duration_s
        return max(0.0, self.w_prime_joules - expenditure)

    def power_curve(self, durations_s: list[float] | None = None) -> tuple[list[float], list[float]]:
        """Generate power-duration curve."""
        if durations_s is None:
            durations_s = [5, 10, 15, 30, 60, 120, 180, 300, 600,
                           900, 1200, 1800, 3600, 5400, 7200]
        powers = [self.max_power_for_duration(d) for d in durations_s]
        return durations_s, powers


def estimate_cp_from_ftp(ftp: float) -> CriticalPowerModel:
    """Estimate CP and W' from FTP using common relationship.

    CP is typically ~95-97% of FTP. W' varies by athlete type
    but 15-25 kJ is typical for trained cyclists.
    """
    cp = ftp * 0.96
    w_prime = 20000.0  # 20 kJ default
    return CriticalPowerModel(cp_watts=cp, w_prime_joules=w_prime)


def fit_cp_model(power1: float, duration1: float,
                 power2: float, duration2: float) -> CriticalPowerModel:
    """Fit CP model from two power-duration data points.

    Uses the linear form: work = CP * t + W'
    """
    work1 = power1 * duration1
    work2 = power2 * duration2
    dt = duration1 - duration2
    if abs(dt) < 1e-6:
        return estimate_cp_from_ftp(power1)
    cp = (work1 - work2) / dt
    w_prime = work1 - cp * duration1
    cp = max(50, cp)
    w_prime = max(1000, w_prime)
    return CriticalPowerModel(cp_watts=cp, w_prime_joules=w_prime)


# ---------------------------------------------------------------------------
# Route optimization (pacing strategy)
# ---------------------------------------------------------------------------

@dataclass
class PacingSegment:
    """Optimal power/speed for a single segment."""
    segment_index: int
    distance_m: float
    grade_pct: float
    optimal_power: float
    speed_kmh: float
    time_s: float
    strategy_note: str


@dataclass
class PacingResult:
    """Result of route optimization."""
    segments: list[PacingSegment]
    total_time_s: float
    avg_power: float
    total_distance_m: float
    avg_speed_kmh: float
    time_saved_s: float
    even_pace_time_s: float


def optimize_pacing(
    rider: RiderParams,
    bike: BikeParams,
    course_segments: list[CourseSegment],
    ftp: float,
    target_avg_power: float | None = None,
) -> PacingResult:
    """Find optimal pacing for a course.

    Uses a simple negative-split strategy:
    - Push harder on climbs (where extra power yields more time savings)
    - Ease off on descents (where aero drag limits gains)
    - Push harder on flat/headwind segments (better power-to-time ratio)
    """
    if target_avg_power is None:
        target_avg_power = ftp * 0.85

    even_power = target_avg_power
    even_total_time = 0.0
    for seg in course_segments:
        course = CourseParams(
            grade_pct=seg.grade_pct, headwind_kmh=seg.headwind_kmh,
            wind_direction_deg=seg.wind_direction_deg,
            elevation_m=seg.elevation_m, temperature_c=seg.temperature_c,
        )
        result = solve_speed(rider, bike, course, power_override=even_power)
        even_total_time += seg.distance_m / max(result.speed_ms, 0.1)

    power_factors: list[float] = []
    for seg in course_segments:
        if seg.grade_pct > 5:
            factor = 1.15
            note = "Hard climb push"
        elif seg.grade_pct > 2:
            factor = 1.08
            note = "Moderate climb push"
        elif seg.grade_pct > 0:
            factor = 1.03
            note = "Slight incline push"
        elif seg.grade_pct > -2:
            factor = 0.98
            note = "Flat/slight descent ease"
        elif seg.grade_pct > -5:
            factor = 0.88
            note = "Descent recovery"
        else:
            factor = 0.75
            note = "Steep descent coast"
        power_factors.append(factor)

    total_weight = sum(
        f * seg.distance_m for f, seg in zip(power_factors, course_segments)
    )
    total_dist = sum(seg.distance_m for seg in course_segments)
    scale = target_avg_power * total_dist / total_weight if total_weight > 0 else 1.0

    pacing_segments: list[PacingSegment] = []
    total_time = 0.0
    total_power_dist = 0.0

    notes = ["Hard climb push", "Moderate climb push", "Slight incline push",
             "Flat/slight descent ease", "Descent recovery", "Steep descent coast"]

    for i, seg in enumerate(course_segments):
        optimal_power = power_factors[i] * scale
        optimal_power = max(50, min(optimal_power, ftp * 1.20))

        course = CourseParams(
            grade_pct=seg.grade_pct, headwind_kmh=seg.headwind_kmh,
            wind_direction_deg=seg.wind_direction_deg,
            elevation_m=seg.elevation_m, temperature_c=seg.temperature_c,
        )
        result = solve_speed(rider, bike, course,
                             power_override=optimal_power)
        time_s = seg.distance_m / max(result.speed_ms, 0.1)

        if seg.grade_pct > 5:
            note = "Hard climb push"
        elif seg.grade_pct > 2:
            note = "Moderate climb push"
        elif seg.grade_pct > 0:
            note = "Slight incline push"
        elif seg.grade_pct > -2:
            note = "Flat/slight descent ease"
        elif seg.grade_pct > -5:
            note = "Descent recovery"
        else:
            note = "Steep descent coast"

        pacing_segments.append(PacingSegment(
            segment_index=i,
            distance_m=seg.distance_m,
            grade_pct=seg.grade_pct,
            optimal_power=optimal_power,
            speed_kmh=result.speed_kmh,
            time_s=time_s,
            strategy_note=note,
        ))
        total_time += time_s
        total_power_dist += optimal_power * seg.distance_m

    avg_power = total_power_dist / total_dist if total_dist > 0 else 0
    avg_speed = (total_dist / total_time * 3.6) if total_time > 0 else 0

    return PacingResult(
        segments=pacing_segments,
        total_time_s=total_time,
        avg_power=avg_power,
        total_distance_m=total_dist,
        avg_speed_kmh=avg_speed,
        time_saved_s=even_total_time - total_time,
        even_pace_time_s=even_total_time,
    )


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def save_leaderboard(entries: list[LeaderboardEntry], filepath: str | Path) -> None:
    """Save leaderboard entries to JSON."""
    data = [
        {
            "course_name": e.course_name,
            "rider_name": e.rider_name,
            "time_s": e.time_s,
            "avg_speed_kmh": e.avg_speed_kmh,
            "power_watts": e.power_watts,
            "date": e.date,
        }
        for e in entries
    ]
    Path(filepath).write_text(json.dumps(data, indent=2))


def load_leaderboard(filepath: str | Path) -> list[LeaderboardEntry]:
    """Load leaderboard entries from JSON."""
    path = Path(filepath)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [LeaderboardEntry(**d) for d in data]


# ---------------------------------------------------------------------------
# Preset save/load
# ---------------------------------------------------------------------------

def _serialize_preset(preset: Preset) -> dict:
    return {
        "name": preset.name,
        "rider": {
            "power_watts": preset.rider.power_watts,
            "weight_kg": preset.rider.weight_kg,
            "height_cm": preset.rider.height_cm,
        },
        "bike": {
            "weight_kg": preset.bike.weight_kg,
            "tire_type": preset.bike.tire_type.value,
            "position": preset.bike.position.value,
            "drivetrain_efficiency": preset.bike.drivetrain_efficiency,
            "wheel_mass_kg": preset.bike.wheel_mass_kg,
            "wheel_radius_m": preset.bike.wheel_radius_m,
        },
        "course": {
            "grade_pct": preset.course.grade_pct,
            "headwind_kmh": preset.course.headwind_kmh,
            "wind_direction_deg": preset.course.wind_direction_deg,
            "elevation_m": preset.course.elevation_m,
            "temperature_c": preset.course.temperature_c,
        },
    }


def _deserialize_preset(data: dict) -> Preset:
    tire = next(t for t in TireType if t.value == data["bike"]["tire_type"])
    pos = next(p for p in RidingPosition if p.value == data["bike"]["position"])
    return Preset(
        name=data["name"],
        rider=RiderParams(**data["rider"]),
        bike=BikeParams(
            weight_kg=data["bike"]["weight_kg"],
            tire_type=tire,
            position=pos,
            drivetrain_efficiency=data["bike"]["drivetrain_efficiency"],
            wheel_mass_kg=data["bike"].get("wheel_mass_kg", DEFAULT_WHEEL_MASS),
            wheel_radius_m=data["bike"].get("wheel_radius_m", DEFAULT_WHEEL_RADIUS),
        ),
        course=CourseParams(**data["course"]),
    )


def save_presets(presets: list[Preset], filepath: str | Path) -> None:
    """Save a list of presets to a JSON file."""
    data = [_serialize_preset(p) for p in presets]
    Path(filepath).write_text(json.dumps(data, indent=2))


def load_presets(filepath: str | Path) -> list[Preset]:
    """Load presets from a JSON file."""
    data = json.loads(Path(filepath).read_text())
    return [_deserialize_preset(d) for d in data]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_result_csv(
    result: SimulationResult,
    rider: RiderParams,
    bike: BikeParams,
    course: CourseParams,
    filepath: str | Path,
) -> None:
    """Export a single simulation result to CSV."""
    lines = [
        "Parameter,Value",
        f"Power (W),{rider.power_watts}",
        f"Rider Weight (kg),{rider.weight_kg}",
        f"Rider Height (cm),{rider.height_cm}",
        f"Bike Weight (kg),{bike.weight_kg}",
        f"Tire Type,{bike.tire_type.value}",
        f"Position,{bike.position.value}",
        f"Drivetrain Efficiency,{bike.drivetrain_efficiency}",
        f"Grade (%),{course.grade_pct}",
        f"Headwind (km/h),{course.headwind_kmh}",
        f"Wind Direction (deg),{course.wind_direction_deg}",
        f"Elevation (m),{course.elevation_m}",
        f"Temperature (C),{course.temperature_c}",
        "",
        "Result,Value",
        f"Speed (km/h),{result.speed_kmh:.2f}",
        f"Speed (mph),{result.speed_mph:.2f}",
        f"Air Density (kg/m3),{result.air_density:.4f}",
        f"CdA (m2),{result.cda:.4f}",
        f"Crr,{result.crr:.4f}",
        f"Power Aero (W),{result.power_aero:.1f}",
        f"Power Rolling (W),{result.power_rolling:.1f}",
        f"Power Gravity (W),{result.power_gravity:.1f}",
        f"Power Drivetrain Loss (W),{result.power_drivetrain_loss:.1f}",
    ]
    Path(filepath).write_text("\n".join(lines))


def export_course_csv(
    profile: CourseProfileResult,
    filepath: str | Path,
) -> None:
    """Export course profile results to CSV."""
    lines = [
        "Segment,Distance (m),Grade (%),Speed (km/h),Time (s),Elev Start (m),Elev End (m),P_Aero (W),P_Roll (W),P_Grav (W)"
    ]
    for seg in profile.segments:
        lines.append(
            f"{seg.segment_index + 1},{seg.distance_m:.0f},{seg.grade_pct:.1f},"
            f"{seg.speed_kmh:.2f},{seg.time_s:.1f},"
            f"{seg.elevation_start_m:.1f},{seg.elevation_end_m:.1f},"
            f"{seg.power_aero:.1f},{seg.power_rolling:.1f},{seg.power_gravity:.1f}"
        )
    lines.append("")
    lines.append(f"Total Distance (m),{profile.total_distance_m:.0f}")
    lines.append(f"Total Time (s),{profile.total_time_s:.1f}")
    lines.append(f"Average Speed (km/h),{profile.avg_speed_kmh:.2f}")
    lines.append(f"Elevation Gain (m),{profile.total_elevation_gain_m:.1f}")
    lines.append(f"Elevation Loss (m),{profile.total_elevation_loss_m:.1f}")
    Path(filepath).write_text("\n".join(lines))


def export_comparison_csv(
    scenarios: list[tuple[str, RiderParams, BikeParams, CourseParams, SimulationResult]],
    filepath: str | Path,
) -> None:
    """Export comparison of multiple scenarios to CSV."""
    lines = [
        "Scenario,Power (W),Rider Wt (kg),Bike Wt (kg),Grade (%),Wind (km/h),"
        "Speed (km/h),Speed (mph),P_Aero (W),P_Roll (W),P_Grav (W)"
    ]
    for name, rider, bike, course, result in scenarios:
        lines.append(
            f"{name},{rider.power_watts},{rider.weight_kg},{bike.weight_kg},"
            f"{course.grade_pct},{course.headwind_kmh},"
            f"{result.speed_kmh:.2f},{result.speed_mph:.2f},"
            f"{result.power_aero:.1f},{result.power_rolling:.1f},{result.power_gravity:.1f}"
        )
    Path(filepath).write_text("\n".join(lines))


def export_workout_csv(
    workout: WorkoutResult,
    filepath: str | Path,
) -> None:
    """Export workout results to CSV."""
    lines = [
        "Time (s),Power (W),Speed (km/h),Distance (m),Calories (kcal)"
    ]
    for pt in workout.points:
        lines.append(
            f"{pt.time_s:.1f},{pt.power_watts:.0f},{pt.speed_kmh:.2f},"
            f"{pt.distance_m:.1f},{pt.calories_kcal:.1f}"
        )
    lines.append("")
    lines.append(f"Total Time (s),{workout.total_time_s:.1f}")
    lines.append(f"Total Distance (m),{workout.total_distance_m:.0f}")
    lines.append(f"Avg Speed (km/h),{workout.avg_speed_kmh:.2f}")
    lines.append(f"Avg Power (W),{workout.avg_power_watts:.0f}")
    lines.append(f"Total Work (kJ),{workout.total_work_kj:.1f}")
    lines.append(f"Total Calories (kcal),{workout.total_calories_kcal:.0f}")
    Path(filepath).write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Built-in presets
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


# ---------------------------------------------------------------------------
# Real-time simulation support
# ---------------------------------------------------------------------------

@dataclass
class RealTimeState:
    """Instantaneous state during a real-time ride simulation."""
    elapsed_time_s: float
    distance_m: float
    current_speed_kmh: float
    current_speed_ms: float
    current_power_watts: float
    current_grade_pct: float
    current_elevation_m: float
    segment_index: int
    segment_progress: float
    w_prime_balance_j: float
    w_prime_max_j: float
    avg_speed_kmh: float
    total_distance_m: float
    calories_kcal: float
    power_zone_name: str
    power_zone_color: str
    is_finished: bool


def find_segment_at_distance(
    segments: list[CourseSegment],
    distance_m: float,
) -> tuple[int, float, float]:
    """Find which segment the rider is in at a given distance.

    Returns (segment_index, distance_into_segment, cumulative_distance_to_segment_start).
    """
    cumulative = 0.0
    for i, seg in enumerate(segments):
        if cumulative + seg.distance_m > distance_m:
            return i, distance_m - cumulative, cumulative
        cumulative += seg.distance_m
    return len(segments) - 1, 0.0, cumulative


def compute_elevation_at_distance(
    segments: list[CourseSegment],
    distance_m: float,
) -> float:
    """Compute the rider's elevation at a given distance along the course."""
    cumulative = 0.0
    elevation = segments[0].elevation_m if segments else 0.0

    for seg in segments:
        seg_end_dist = cumulative + seg.distance_m
        if distance_m <= seg_end_dist:
            dist_into = distance_m - cumulative
            elev_change = dist_into * math.sin(grade_to_radians(seg.grade_pct))
            return elevation + elev_change
        elev_change = seg.distance_m * math.sin(grade_to_radians(seg.grade_pct))
        elevation += elev_change
        cumulative = seg_end_dist

    return elevation


def compute_realtime_tick(
    rider: RiderParams,
    bike: BikeParams,
    segments: list[CourseSegment],
    power_watts: float,
    ftp: float,
    cp_model: CriticalPowerModel | None,
    elapsed_time_s: float,
    dt: float,
    prev_distance_m: float,
    prev_w_prime_j: float,
    prev_calories: float,
) -> RealTimeState:
    """Advance a real-time ride simulation by one time step.

    The rider's speed is solved from the current segment's conditions
    and the supplied power_watts (which can change each tick).
    """
    total_distance = sum(seg.distance_m for seg in segments)

    seg_idx, dist_into_seg, cum_to_seg = find_segment_at_distance(
        segments, min(prev_distance_m, total_distance - 0.01)
    )
    seg = segments[seg_idx]
    seg_progress = dist_into_seg / seg.distance_m if seg.distance_m > 0 else 1.0

    current_elevation = compute_elevation_at_distance(segments, prev_distance_m)

    course = CourseParams(
        grade_pct=seg.grade_pct,
        headwind_kmh=seg.headwind_kmh,
        wind_direction_deg=seg.wind_direction_deg,
        elevation_m=current_elevation,
        temperature_c=seg.temperature_c,
    )
    result = solve_speed(rider, bike, course, power_override=power_watts)

    new_distance = prev_distance_m + result.speed_ms * dt
    is_finished = new_distance >= total_distance
    new_distance = min(new_distance, total_distance)

    # W' balance tracking
    if cp_model is not None:
        if power_watts > cp_model.cp_watts:
            w_prime_spent = (power_watts - cp_model.cp_watts) * dt
            new_w_prime = max(0.0, prev_w_prime_j - w_prime_spent)
        else:
            tau = 546.0 * math.exp(-0.01 * (cp_model.cp_watts - power_watts)) + 316.0
            recovery = (cp_model.w_prime_joules - prev_w_prime_j) * (1 - math.exp(-dt / tau))
            new_w_prime = min(cp_model.w_prime_joules, prev_w_prime_j + recovery)
        w_prime_max = cp_model.w_prime_joules
    else:
        new_w_prime = prev_w_prime_j
        w_prime_max = prev_w_prime_j

    new_calories = prev_calories + estimate_calories(power_watts, dt)

    zone_name, zone_color = zone_for_power(power_watts, ftp)

    new_elapsed = elapsed_time_s + dt
    avg_speed = (new_distance / new_elapsed * 3.6) if new_elapsed > 0 else 0.0

    return RealTimeState(
        elapsed_time_s=new_elapsed,
        distance_m=new_distance,
        current_speed_kmh=result.speed_kmh,
        current_speed_ms=result.speed_ms,
        current_power_watts=power_watts,
        current_grade_pct=seg.grade_pct,
        current_elevation_m=current_elevation,
        segment_index=seg_idx,
        segment_progress=seg_progress,
        w_prime_balance_j=new_w_prime,
        w_prime_max_j=w_prime_max,
        avg_speed_kmh=avg_speed,
        total_distance_m=total_distance,
        calories_kcal=new_calories,
        power_zone_name=zone_name,
        power_zone_color=zone_color,
        is_finished=is_finished,
    )


# ---------------------------------------------------------------------------
# GPX ride analysis — predicted vs actual speed
# ---------------------------------------------------------------------------

@dataclass
class GPXRidePoint:
    """A point from a parsed GPX ride with time, position, and optional power."""
    time_s: float
    distance_m: float
    elevation_m: float
    speed_kmh: float
    grade_pct: float
    lat: float
    lon: float
    power_watts: float | None = None


@dataclass
class RideAnalysis:
    """Result of comparing predicted vs actual ride data."""
    points: list[GPXRidePoint]
    predicted_speeds: list[float]
    actual_speeds: list[float]
    distances_km: list[float]
    elevations: list[float]
    grades: list[float]
    mean_error_kmh: float
    rmse_kmh: float
    mean_pct_error: float
    total_distance_m: float
    total_time_s: float
    actual_avg_speed_kmh: float
    predicted_avg_speed_kmh: float


def parse_gpx_ride(filepath: str | Path) -> list[GPXRidePoint]:
    """Parse a GPX file with time data into ride points.

    Extracts timestamps, computes speed between points, and optionally
    reads power data from Garmin TrackPointExtension.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    trackpoints: list[tuple[float, float, float, str | None, float | None]] = []
    for trkpt in root.iter(f"{ns}trkpt"):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])
        ele_elem = trkpt.find(f"{ns}ele")
        ele = float(ele_elem.text) if ele_elem is not None else 0.0

        time_elem = trkpt.find(f"{ns}time")
        time_str = time_elem.text if time_elem is not None else None

        # Try to find power in extensions
        power_val: float | None = None
        extensions = trkpt.find(f"{ns}extensions")
        if extensions is not None:
            # Garmin TrackPointExtension
            for child in extensions:
                tag_lower = child.tag.lower()
                if "power" in tag_lower:
                    try:
                        power_val = float(child.text)
                    except (TypeError, ValueError):
                        pass
                # Also check nested extension elements
                for sub in child:
                    sub_tag = sub.tag.lower()
                    if "power" in sub_tag or "watts" in sub_tag:
                        try:
                            power_val = float(sub.text)
                        except (TypeError, ValueError):
                            pass

        trackpoints.append((lat, lon, ele, time_str, power_val))

    if len(trackpoints) < 2:
        return []

    # Parse timestamps
    import re
    def _parse_iso(s: str | None) -> float | None:
        if not s:
            return None
        s = s.strip()
        s = re.sub(r'\.\d+', '', s)
        try:
            import datetime
            dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError):
            return None

    base_ts = _parse_iso(trackpoints[0][3])
    if base_ts is None:
        return []

    points: list[GPXRidePoint] = []
    cum_dist = 0.0

    for i in range(len(trackpoints)):
        ts = _parse_iso(trackpoints[i][3])
        if ts is None:
            continue
        time_s = ts - base_ts

        if i == 0:
            speed = 0.0
            grade = 0.0
        else:
            dist = _haversine(
                trackpoints[i - 1][0], trackpoints[i - 1][1],
                trackpoints[i][0], trackpoints[i][1],
            )
            dt = time_s - ((_parse_iso(trackpoints[i - 1][3]) or base_ts) - base_ts)
            cum_dist += dist
            speed = (dist / dt * 3.6) if dt > 0 else 0.0
            grade = (trackpoints[i][2] - trackpoints[i - 1][2]) / dist * 100.0 if dist > 0 else 0.0

        points.append(GPXRidePoint(
            time_s=time_s,
            distance_m=cum_dist,
            elevation_m=trackpoints[i][2],
            speed_kmh=min(speed, 120.0),  # Cap unrealistic speeds
            grade_pct=max(-30, min(30, grade)),
            lat=trackpoints[i][0],
            lon=trackpoints[i][1],
            power_watts=trackpoints[i][4],
        ))

    return points


def analyze_ride(
    points: list[GPXRidePoint],
    rider: RiderParams,
    bike: BikeParams,
    smoothing_window: int = 5,
    temperature_c: float = 20.0,
    headwind_kmh: float = 0.0,
    wind_direction_deg: float = 0.0,
) -> RideAnalysis:
    """Compare actual ride data against model predictions.

    For each point, predicts speed using the simulation model at the
    point's grade and elevation, then computes error metrics.
    Uses the same physics as Live Ride (solve_speed with power_override).
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 ride points for analysis")

    # Smooth grade data to reduce GPS noise
    grades = [p.grade_pct for p in points]
    smoothed_grades: list[float] = []
    for i in range(len(grades)):
        lo = max(0, i - smoothing_window // 2)
        hi = min(len(grades), i + smoothing_window // 2 + 1)
        smoothed_grades.append(sum(grades[lo:hi]) / (hi - lo))

    actual_speeds: list[float] = []
    predicted_speeds: list[float] = []
    distances_km: list[float] = []
    elevations: list[float] = []

    for i, pt in enumerate(points):
        if pt.speed_kmh < 1.0:
            continue  # Skip stopped points

        power = pt.power_watts if pt.power_watts is not None else rider.power_watts
        course = CourseParams(
            grade_pct=smoothed_grades[i],
            elevation_m=pt.elevation_m,
            temperature_c=temperature_c,
            headwind_kmh=headwind_kmh,
            wind_direction_deg=wind_direction_deg,
        )
        pred = solve_speed(rider, bike, course, power_override=power)

        actual_speeds.append(pt.speed_kmh)
        predicted_speeds.append(pred.speed_kmh)
        distances_km.append(pt.distance_m / 1000.0)
        elevations.append(pt.elevation_m)

    if not actual_speeds:
        raise ValueError("No valid data points found in ride")

    # Error metrics
    errors = [p - a for p, a in zip(predicted_speeds, actual_speeds)]
    mean_error = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
    pct_errors = [abs(p - a) / max(a, 1.0) * 100.0
                  for p, a in zip(predicted_speeds, actual_speeds)]
    mean_pct = sum(pct_errors) / len(pct_errors)

    total_dist = points[-1].distance_m
    total_time = points[-1].time_s
    actual_avg = total_dist / total_time * 3.6 if total_time > 0 else 0.0
    pred_avg = sum(predicted_speeds) / len(predicted_speeds) if predicted_speeds else 0.0

    return RideAnalysis(
        points=points,
        predicted_speeds=predicted_speeds,
        actual_speeds=actual_speeds,
        distances_km=distances_km,
        elevations=elevations,
        grades=smoothed_grades[:len(distances_km)],
        mean_error_kmh=mean_error,
        rmse_kmh=rmse,
        mean_pct_error=mean_pct,
        total_distance_m=total_dist,
        total_time_s=total_time,
        actual_avg_speed_kmh=actual_avg,
        predicted_avg_speed_kmh=pred_avg,
    )


# ---------------------------------------------------------------------------
# What-if scenario analysis
# ---------------------------------------------------------------------------

@dataclass
class WhatIfChange:
    """A single parameter change for what-if analysis."""
    parameter: str  # e.g. "weight_kg", "power_watts", "position", "tire_type"
    label: str
    original_value: Any
    new_value: Any


@dataclass
class WhatIfResult:
    """Result of a what-if scenario on a course."""
    change: WhatIfChange
    original_time_s: float
    new_time_s: float
    time_diff_s: float
    original_avg_speed_kmh: float
    new_avg_speed_kmh: float
    speed_diff_kmh: float
    pct_improvement: float


def what_if_analysis(
    rider: RiderParams,
    bike: BikeParams,
    segments: list[CourseSegment],
    changes: list[WhatIfChange],
) -> list[WhatIfResult]:
    """Evaluate time impact of each parameter change on a course."""
    base_result = simulate_course_profile(rider, bike, segments)
    base_time = base_result.total_time_s
    base_speed = base_result.avg_speed_kmh

    results: list[WhatIfResult] = []

    for change in changes:
        test_rider = RiderParams(
            power_watts=rider.power_watts,
            weight_kg=rider.weight_kg,
            height_cm=rider.height_cm,
        )
        test_bike = BikeParams(
            weight_kg=bike.weight_kg,
            tire_type=bike.tire_type,
            position=bike.position,
            drivetrain_efficiency=bike.drivetrain_efficiency,
            wheel_mass_kg=bike.wheel_mass_kg,
            wheel_radius_m=bike.wheel_radius_m,
        )

        param = change.parameter
        val = change.new_value

        if param == "weight_kg":
            test_rider.weight_kg = float(val)
        elif param == "power_watts":
            test_rider.power_watts = float(val)
        elif param == "height_cm":
            test_rider.height_cm = float(val)
        elif param == "bike_weight_kg":
            test_bike.weight_kg = float(val)
        elif param == "position":
            test_bike.position = val
        elif param == "tire_type":
            test_bike.tire_type = val
        elif param == "drivetrain_efficiency":
            test_bike.drivetrain_efficiency = float(val)

        new_result = simulate_course_profile(test_rider, test_bike, segments)
        new_time = new_result.total_time_s
        new_speed = new_result.avg_speed_kmh
        time_diff = base_time - new_time  # positive = faster
        speed_diff = new_speed - base_speed  # positive = faster
        pct = (time_diff / base_time * 100.0) if base_time > 0 else 0.0

        results.append(WhatIfResult(
            change=change,
            original_time_s=base_time,
            new_time_s=new_time,
            time_diff_s=time_diff,
            original_avg_speed_kmh=base_speed,
            new_avg_speed_kmh=new_speed,
            speed_diff_kmh=speed_diff,
            pct_improvement=pct,
        ))

    return results


# ---------------------------------------------------------------------------
# Segment KOM predictor
# ---------------------------------------------------------------------------

# W/kg benchmarks for KOM categories at different gradients
# Format: (category_name, w_kg_flat, w_kg_5pct, w_kg_8pct)
KOM_BENCHMARKS: list[tuple[str, float, float, float]] = [
    ("World Tour Pro", 6.2, 6.5, 6.8),
    ("Domestic Pro", 5.5, 5.8, 6.0),
    ("Cat 1", 5.0, 5.2, 5.4),
    ("Cat 2", 4.3, 4.5, 4.7),
    ("Cat 3", 3.7, 3.9, 4.1),
    ("Cat 4", 3.0, 3.2, 3.4),
    ("Cat 5", 2.3, 2.5, 2.7),
    ("Beginner", 1.5, 1.7, 1.9),
]


@dataclass
class SegmentKOMResult:
    """KOM prediction for a single segment."""
    segment_index: int
    distance_m: float
    grade_pct: float
    elevation_gain_m: float
    rider_time_s: float
    rider_speed_kmh: float
    rider_power_watts: float
    rider_w_kg: float
    category_times: list[tuple[str, float, float]]  # (name, time_s, speed_kmh)
    rider_rank: str  # Which category bracket the rider falls into


@dataclass
class KOMPrediction:
    """Full KOM prediction result for a course."""
    segments: list[SegmentKOMResult]
    total_rider_time_s: float
    total_category_times: list[tuple[str, float]]  # (name, total_time_s)


def predict_kom(
    rider: RiderParams,
    bike: BikeParams,
    segments: list[CourseSegment],
) -> KOMPrediction:
    """Predict segment times and compare against category benchmarks."""
    rider_weight = rider.weight_kg + bike.weight_kg
    rider_w_kg = rider.power_watts / rider.weight_kg if rider.weight_kg > 0 else 0

    segment_results: list[SegmentKOMResult] = []
    total_rider_time = 0.0
    cat_total_times: dict[str, float] = {cat[0]: 0.0 for cat in KOM_BENCHMARKS}

    for i, seg in enumerate(segments):
        # Rider's time
        course = CourseParams(
            grade_pct=seg.grade_pct,
            headwind_kmh=seg.headwind_kmh,
            wind_direction_deg=seg.wind_direction_deg,
            elevation_m=seg.elevation_m,
            temperature_c=seg.temperature_c,
        )
        rider_result = solve_speed(rider, bike, course)
        rider_time = seg.distance_m / max(rider_result.speed_ms, 0.1)
        total_rider_time += rider_time

        elev_gain = max(0, seg.distance_m * math.sin(grade_to_radians(seg.grade_pct)))

        # Interpolate W/kg benchmark based on segment grade
        def _interp_wkg(flat: float, mid: float, steep: float, grade: float) -> float:
            abs_g = abs(grade)
            if abs_g <= 0:
                return flat
            elif abs_g <= 5:
                return flat + (mid - flat) * (abs_g / 5.0)
            elif abs_g <= 8:
                return mid + (steep - mid) * ((abs_g - 5.0) / 3.0)
            else:
                return steep

        # Category times
        cat_times: list[tuple[str, float, float]] = []
        for cat_name, wkg_flat, wkg_5, wkg_8 in KOM_BENCHMARKS:
            cat_wkg = _interp_wkg(wkg_flat, wkg_5, wkg_8, seg.grade_pct)
            cat_power = cat_wkg * 75.0  # Reference weight
            cat_rider = RiderParams(power_watts=cat_power, weight_kg=75.0, height_cm=178.0)
            cat_result = solve_speed(cat_rider, bike, course)
            cat_time = seg.distance_m / max(cat_result.speed_ms, 0.1)
            cat_speed = cat_result.speed_kmh
            cat_times.append((cat_name, cat_time, cat_speed))
            cat_total_times[cat_name] += cat_time

        # Determine rider's category rank
        rank = "Beginner"
        for cat_name, cat_time, _ in cat_times:
            if rider_time <= cat_time:
                rank = cat_name
                break

        segment_results.append(SegmentKOMResult(
            segment_index=i,
            distance_m=seg.distance_m,
            grade_pct=seg.grade_pct,
            elevation_gain_m=elev_gain,
            rider_time_s=rider_time,
            rider_speed_kmh=rider_result.speed_kmh,
            rider_power_watts=rider.power_watts,
            rider_w_kg=rider_w_kg,
            category_times=cat_times,
            rider_rank=rank,
        ))

    total_cat = [(name, cat_total_times[name]) for name, _, _, _ in KOM_BENCHMARKS]

    return KOMPrediction(
        segments=segment_results,
        total_rider_time_s=total_rider_time,
        total_category_times=total_cat,
    )


# ---------------------------------------------------------------------------
# Race Planner: Monte Carlo, parameter sweeps & optimization
# ---------------------------------------------------------------------------

# Canonical planner variables and their (metric) display metadata.
# ``kind`` is "rider"/"bike"/"course"/"aero"/"roll" — used only for routing.
PLANNER_PARAMS: dict[str, dict] = {
    "power_watts": {"label": "Avg Power", "unit": "W", "kind": "rider"},
    "weight_kg": {"label": "Rider Weight", "unit": "kg", "kind": "rider"},
    "bike_weight_kg": {"label": "Bike Weight", "unit": "kg", "kind": "bike"},
    "headwind_kmh": {"label": "Wind Speed", "unit": "km/h", "kind": "course"},
    "wind_direction_deg": {"label": "Wind Direction", "unit": "\u00b0", "kind": "course"},
    "temperature_c": {"label": "Temperature", "unit": "\u00b0C", "kind": "course"},
    "cda": {"label": "Aero (CdA)", "unit": "m\u00b2", "kind": "aero"},
    "crr": {"label": "Rolling (Crr)", "unit": "", "kind": "roll"},
}


def _clamp_planner_value(key: str, value: float) -> float:
    """Keep a sampled/swept planner value physically valid."""
    if key == "power_watts":
        return max(1.0, value)
    if key == "weight_kg":
        return max(20.0, value)
    if key == "bike_weight_kg":
        return max(1.0, value)
    if key == "cda":
        return max(0.05, value)
    if key == "crr":
        return max(0.0005, value)
    if key == "temperature_c":
        return max(-30.0, min(60.0, value))
    if key == "wind_direction_deg":
        return value % 360.0
    return value


def planner_base_value(
    rider: RiderParams,
    bike: BikeParams,
    segments: list[CourseSegment],
    key: str,
) -> float:
    """Return the current value of a planner variable for the given setup.

    Course-level variables (wind, temperature) are averaged across segments.
    """
    if key == "power_watts":
        return rider.power_watts
    if key == "weight_kg":
        return rider.weight_kg
    if key == "bike_weight_kg":
        return bike.weight_kg
    if key == "cda":
        return calculate_cda(rider.height_cm, bike.position)
    if key == "crr":
        return TIRE_CRR[bike.tire_type]
    if not segments:
        return 0.0
    if key == "headwind_kmh":
        return sum(s.headwind_kmh for s in segments) / len(segments)
    if key == "wind_direction_deg":
        return sum(s.wind_direction_deg for s in segments) / len(segments)
    if key == "temperature_c":
        return sum(s.temperature_c for s in segments) / len(segments)
    return 0.0


def _planner_evaluate(
    rider: RiderParams,
    bike: BikeParams,
    segments: list[CourseSegment],
    overrides: dict[str, float],
) -> tuple[float, float]:
    """Evaluate total time (s) and avg speed (km/h) with parameter overrides.

    ``overrides`` maps planner keys to absolute values. Missing keys keep
    each segment's own value (for course variables) or the base rider/bike
    value. Rider/bike are copied so the caller's objects are untouched.
    """
    test_rider = replace(rider)
    test_bike = replace(bike)
    if "power_watts" in overrides:
        test_rider.power_watts = overrides["power_watts"]
    if "weight_kg" in overrides:
        test_rider.weight_kg = overrides["weight_kg"]
    if "bike_weight_kg" in overrides:
        test_bike.weight_kg = overrides["bike_weight_kg"]

    cda_override = overrides.get("cda")
    crr_override = overrides.get("crr")

    total_time = 0.0
    total_dist = 0.0
    for seg in segments:
        course = CourseParams(
            grade_pct=seg.grade_pct,
            headwind_kmh=overrides.get("headwind_kmh", seg.headwind_kmh),
            wind_direction_deg=overrides.get("wind_direction_deg", seg.wind_direction_deg),
            elevation_m=seg.elevation_m,
            temperature_c=overrides.get("temperature_c", seg.temperature_c),
        )
        res = solve_speed(
            test_rider, test_bike, course,
            cda_override=cda_override, crr_override=crr_override,
        )
        total_time += seg.distance_m / max(res.speed_ms, 0.1)
        total_dist += seg.distance_m

    avg_speed = (total_dist / total_time * 3.6) if total_time > 0 else 0.0
    return total_time, avg_speed


# ---- Monte Carlo -----------------------------------------------------------

@dataclass
class MCVariable:
    """A single uncertain input for a Monte Carlo race simulation."""
    key: str
    distribution: str = "normal"  # "normal" | "uniform"
    mean: float = 0.0
    std: float = 0.0
    low: float = 0.0
    high: float = 0.0


@dataclass
class MonteCarloResult:
    """Distribution of finish times from a Monte Carlo simulation."""
    n: int
    distance_m: float
    times_s: list[float]
    avg_speeds_kmh: list[float]
    mean_time_s: float
    std_time_s: float
    percentiles_s: dict[int, float]
    min_time_s: float
    max_time_s: float
    variable_keys: list[str]


def monte_carlo_simulation(
    rider: RiderParams,
    bike: BikeParams,
    segments: list[CourseSegment],
    variables: list[MCVariable],
    n: int = 1000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run ``n`` random course simulations, sampling the given variables.

    Returns the full distribution of finish times plus summary statistics
    (mean/std and P5/P10/P25/P50/P75/P90/P95 percentiles).
    """
    if not segments:
        raise ValueError("Monte Carlo requires at least one course segment.")
    if n < 1:
        raise ValueError("n must be >= 1.")

    rng = np.random.default_rng(seed)
    times: list[float] = []
    speeds: list[float] = []
    distance = sum(s.distance_m for s in segments)

    for _ in range(n):
        overrides: dict[str, float] = {}
        for v in variables:
            if v.distribution == "uniform":
                val = rng.uniform(v.low, v.high)
            else:
                val = rng.normal(v.mean, v.std)
            overrides[v.key] = _clamp_planner_value(v.key, float(val))
        t, s = _planner_evaluate(rider, bike, segments, overrides)
        times.append(t)
        speeds.append(s)

    arr = np.array(times)
    pct_levels = [5, 10, 25, 50, 75, 90, 95]
    percentiles = {p: float(np.percentile(arr, p)) for p in pct_levels}

    return MonteCarloResult(
        n=n,
        distance_m=distance,
        times_s=times,
        avg_speeds_kmh=speeds,
        mean_time_s=float(arr.mean()),
        std_time_s=float(arr.std()),
        percentiles_s=percentiles,
        min_time_s=float(arr.min()),
        max_time_s=float(arr.max()),
        variable_keys=[v.key for v in variables],
    )


# ---- 1D parameter sweep ----------------------------------------------------

@dataclass
class Sweep1DResult:
    """Finish-time response to sweeping a single input."""
    key: str
    values: list[float]
    times_s: list[float]
    avg_speeds_kmh: list[float]
    distance_m: float
    base_value: float
    base_time_s: float


def sweep_1d(
    rider: RiderParams,
    bike: BikeParams,
    segments: list[CourseSegment],
    key: str,
    min_value: float,
    max_value: float,
    num_points: int = 25,
) -> Sweep1DResult:
    """Sweep one variable across [min, max] and record finish time/speed."""
    if not segments:
        raise ValueError("Sweep requires at least one course segment.")
    if key not in PLANNER_PARAMS:
        raise ValueError(f"Unknown planner variable: {key}")
    num_points = max(2, num_points)

    values = list(np.linspace(min_value, max_value, num_points))
    times: list[float] = []
    speeds: list[float] = []
    for val in values:
        v = _clamp_planner_value(key, float(val))
        t, s = _planner_evaluate(rider, bike, segments, {key: v})
        times.append(t)
        speeds.append(s)

    base_value = planner_base_value(rider, bike, segments, key)
    base_time, _ = _planner_evaluate(rider, bike, segments, {})

    return Sweep1DResult(
        key=key,
        values=values,
        times_s=times,
        avg_speeds_kmh=speeds,
        distance_m=sum(s.distance_m for s in segments),
        base_value=base_value,
        base_time_s=base_time,
    )


# ---- 2D parameter sweep (contour) ------------------------------------------

@dataclass
class Sweep2DResult:
    """Finish-time grid over two swept inputs (for a contour/heatmap)."""
    key_x: str
    key_y: str
    x_values: list[float]
    y_values: list[float]
    times_s: list[list[float]]  # times_s[iy][ix]
    distance_m: float
    best_x: float
    best_y: float
    best_time_s: float
    base_x: float
    base_y: float


def sweep_2d(
    rider: RiderParams,
    bike: BikeParams,
    segments: list[CourseSegment],
    key_x: str,
    key_y: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    num_x: int = 25,
    num_y: int = 25,
) -> Sweep2DResult:
    """Sweep two variables on a grid; return a time matrix + optimum cell."""
    if not segments:
        raise ValueError("Sweep requires at least one course segment.")
    if key_x not in PLANNER_PARAMS or key_y not in PLANNER_PARAMS:
        raise ValueError("Unknown planner variable.")
    if key_x == key_y:
        raise ValueError("Choose two different variables for a 2D sweep.")
    num_x = max(2, num_x)
    num_y = max(2, num_y)

    x_values = list(np.linspace(x_range[0], x_range[1], num_x))
    y_values = list(np.linspace(y_range[0], y_range[1], num_y))

    grid: list[list[float]] = []
    best_time = float("inf")
    best_x = x_values[0]
    best_y = y_values[0]

    for yv in y_values:
        row: list[float] = []
        yc = _clamp_planner_value(key_y, float(yv))
        for xv in x_values:
            xc = _clamp_planner_value(key_x, float(xv))
            t, _ = _planner_evaluate(
                rider, bike, segments, {key_x: xc, key_y: yc}
            )
            row.append(t)
            if t < best_time:
                best_time = t
                best_x = float(xv)
                best_y = float(yv)
        grid.append(row)

    return Sweep2DResult(
        key_x=key_x,
        key_y=key_y,
        x_values=x_values,
        y_values=y_values,
        times_s=grid,
        distance_m=sum(s.distance_m for s in segments),
        best_x=best_x,
        best_y=best_y,
        best_time_s=best_time,
        base_x=planner_base_value(rider, bike, segments, key_x),
        base_y=planner_base_value(rider, bike, segments, key_y),
    )


# ---- Optimization ----------------------------------------------------------

@dataclass
class OptimizeVariable:
    """A variable to optimize, with inclusive bounds."""
    key: str
    low: float
    high: float


@dataclass
class OptimizationResult:
    """Best achievable setup found within the given bounds."""
    baseline_time_s: float
    optimized_time_s: float
    time_saved_s: float
    baseline_values: dict[str, float]
    optimized_values: dict[str, float]
    distance_m: float
    variable_keys: list[str]
    pacing: "PacingResult | None" = None
    # Finish time of the recommended pacing plan, evaluated with the
    # *optimized* setup (so it is directly comparable to optimized_time_s).
    pacing_time_s: float | None = None


def _golden_section_min(f, a: float, b: float, tol: float = 1e-3,
                        max_iter: int = 80) -> float:
    """Return x in [a, b] minimizing the unimodal function f."""
    if a > b:
        a, b = b, a
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def optimize_setup(
    rider: RiderParams,
    bike: BikeParams,
    segments: list[CourseSegment],
    variables: list[OptimizeVariable],
    ftp: float | None = None,
    include_pacing: bool = True,
    passes: int = 3,
) -> OptimizationResult:
    """Minimize finish time over the chosen variables within their bounds.

    Uses coordinate descent with golden-section search on each variable.
    Optionally returns a recommended pacing plan for the optimized power.
    """
    if not segments:
        raise ValueError("Optimization requires at least one course segment.")

    baseline_time, _ = _planner_evaluate(rider, bike, segments, {})
    baseline_values = {
        v.key: planner_base_value(rider, bike, segments, v.key) for v in variables
    }

    # Start each variable at its base value clamped into the allowed bounds.
    current: dict[str, float] = {}
    for v in variables:
        base = baseline_values[v.key]
        lo, hi = min(v.low, v.high), max(v.low, v.high)
        current[v.key] = _clamp_planner_value(v.key, min(max(base, lo), hi))

    def eval_time(overrides: dict[str, float]) -> float:
        t, _ = _planner_evaluate(rider, bike, segments, overrides)
        return t

    for _ in range(max(1, passes)):
        for v in variables:
            lo, hi = min(v.low, v.high), max(v.low, v.high)

            def f(x: float, key: str = v.key) -> float:
                trial = dict(current)
                trial[key] = _clamp_planner_value(key, x)
                return eval_time(trial)

            best_x = _golden_section_min(f, lo, hi)
            # Compare against the bounds too (optimum is often at an edge).
            candidates = [best_x, lo, hi]
            best = min(candidates, key=lambda x, kk=v.key: f(x, kk))
            current[v.key] = _clamp_planner_value(v.key, best)

    optimized_time = eval_time(current)

    pacing = None
    if include_pacing:
        opt_rider = replace(rider)
        opt_bike = replace(bike)
        if "power_watts" in current:
            opt_rider.power_watts = current["power_watts"]
        if "weight_kg" in current:
            opt_rider.weight_kg = current["weight_kg"]
        if "bike_weight_kg" in current:
            opt_bike.weight_kg = current["bike_weight_kg"]
        base_ftp = ftp if ftp is not None else opt_rider.power_watts
        # Ensure the pacing power cap (ftp*1.2) leaves headroom above the
        # optimized average power, otherwise every segment saturates and the
        # plan comes out flat.
        pacing_ftp = max(base_ftp, opt_rider.power_watts)
        try:
            pacing = optimize_pacing(
                opt_rider, opt_bike, segments, ftp=pacing_ftp,
                target_avg_power=opt_rider.power_watts,
            )
        except Exception:
            pacing = None

    # Re-evaluate the pacing plan's finish time with the full optimized setup
    # (including CdA/Crr/course overrides) so it is comparable to the
    # constant-power optimized time. optimize_pacing's own time ignores those.
    pacing_time = None
    if pacing is not None and pacing.segments:
        cda_over = current.get("cda")
        crr_over = current.get("crr")
        pacing_time = 0.0
        for seg, ps in zip(segments, pacing.segments):
            course = CourseParams(
                grade_pct=seg.grade_pct,
                headwind_kmh=current.get("headwind_kmh", seg.headwind_kmh),
                wind_direction_deg=current.get("wind_direction_deg",
                                               seg.wind_direction_deg),
                elevation_m=seg.elevation_m,
                temperature_c=current.get("temperature_c", seg.temperature_c),
            )
            res = solve_speed(
                opt_rider, opt_bike, course,
                power_override=ps.optimal_power,
                cda_override=cda_over, crr_override=crr_over,
            )
            pacing_time += seg.distance_m / max(res.speed_ms, 0.1)

    return OptimizationResult(
        baseline_time_s=baseline_time,
        optimized_time_s=optimized_time,
        time_saved_s=baseline_time - optimized_time,
        baseline_values=baseline_values,
        optimized_values=current,
        distance_m=sum(s.distance_m for s in segments),
        variable_keys=[v.key for v in variables],
        pacing=pacing,
        pacing_time_s=pacing_time,
    )
