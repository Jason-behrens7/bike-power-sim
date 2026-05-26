"""Bike power simulation physics engine.

Calculates cycling speed from power output, rider/bike parameters,
and environmental conditions using iterative numerical methods.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
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
    max_iter: int = 100,
    tol: float = 1e-6,
) -> SimulationResult:
    """Solve for steady-state speed given power and conditions."""
    pw = power_override if power_override is not None else rider.power_watts
    total_mass = rider.weight_kg + bike.weight_kg
    inertia_factor = wheel_inertia_factor(
        bike.wheel_mass_kg, bike.wheel_radius_m, total_mass
    )
    effective_mass = total_mass * inertia_factor
    cda = calculate_cda(rider.height_cm, bike.position)
    crr = TIRE_CRR[bike.tire_type]
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
