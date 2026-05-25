"""Unit tests for the bike power simulation engine."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from simulation import (
    BikeParams,
    CourseParams,
    CourseSegment,
    Preset,
    RiderParams,
    RidingPosition,
    TireType,
    air_density,
    calculate_cda,
    effective_headwind,
    estimate_frontal_area,
    export_comparison_csv,
    export_course_csv,
    export_result_csv,
    grade_to_radians,
    load_presets,
    resistive_forces,
    save_presets,
    simulate_course_profile,
    solve_speed,
    speed_vs_power_curve,
    validate_all,
    validate_param,
    wheel_inertia_factor,
    # Unit conversions
    kg_to_lbs, lbs_to_kg,
    cm_to_inches, inches_to_cm,
    m_to_feet, feet_to_m,
    celsius_to_fahrenheit, fahrenheit_to_celsius,
    kmh_to_mph, mph_to_kmh,
)


class TestAirDensity(unittest.TestCase):
    def test_sea_level_standard(self) -> None:
        rho = air_density(0, 15)
        self.assertAlmostEqual(rho, 1.225, places=2)

    def test_decreases_with_elevation(self) -> None:
        rho_low = air_density(0, 20)
        rho_high = air_density(2000, 20)
        self.assertGreater(rho_low, rho_high)

    def test_decreases_with_temperature(self) -> None:
        rho_cold = air_density(0, 0)
        rho_hot = air_density(0, 40)
        self.assertGreater(rho_cold, rho_hot)


class TestFrontalArea(unittest.TestCase):
    def test_reasonable_range(self) -> None:
        area = estimate_frontal_area(178)
        self.assertGreater(area, 0.3)
        self.assertLess(area, 0.7)

    def test_taller_larger_area(self) -> None:
        short = estimate_frontal_area(160)
        tall = estimate_frontal_area(195)
        self.assertGreater(tall, short)


class TestCdA(unittest.TestCase):
    def test_aero_position_lower(self) -> None:
        cda_aero = calculate_cda(178, RidingPosition.AEROBARS)
        cda_hoods = calculate_cda(178, RidingPosition.HOODS)
        cda_upright = calculate_cda(178, RidingPosition.UPRIGHT)
        self.assertLess(cda_aero, cda_hoods)
        self.assertLess(cda_hoods, cda_upright)


class TestGradeConversion(unittest.TestCase):
    def test_zero_grade(self) -> None:
        self.assertAlmostEqual(grade_to_radians(0), 0.0)

    def test_positive_grade(self) -> None:
        rad = grade_to_radians(10)
        self.assertAlmostEqual(rad, math.atan(0.1), places=10)

    def test_negative_grade(self) -> None:
        rad = grade_to_radians(-5)
        self.assertLess(rad, 0)


class TestResistiveForces(unittest.TestCase):
    def test_aero_increases_with_speed(self) -> None:
        f_slow = resistive_forces(5, 80, 0.35, 0.004, 1.225, 0, 0)
        f_fast = resistive_forces(15, 80, 0.35, 0.004, 1.225, 0, 0)
        self.assertGreater(f_fast.aero, f_slow.aero)

    def test_rolling_constant_with_speed(self) -> None:
        f1 = resistive_forces(5, 80, 0.35, 0.004, 1.225, 0, 0)
        f2 = resistive_forces(15, 80, 0.35, 0.004, 1.225, 0, 0)
        self.assertAlmostEqual(f1.rolling, f2.rolling, places=6)

    def test_gravity_zero_on_flat(self) -> None:
        f = resistive_forces(10, 80, 0.35, 0.004, 1.225, 0, 0)
        self.assertAlmostEqual(f.gravity, 0.0, places=6)

    def test_gravity_positive_uphill(self) -> None:
        grade_rad = grade_to_radians(5)
        f = resistive_forces(10, 80, 0.35, 0.004, 1.225, grade_rad, 0)
        self.assertGreater(f.gravity, 0)

    def test_gravity_negative_downhill(self) -> None:
        grade_rad = grade_to_radians(-5)
        f = resistive_forces(10, 80, 0.35, 0.004, 1.225, grade_rad, 0)
        self.assertLess(f.gravity, 0)

    def test_headwind_increases_aero(self) -> None:
        f_no_wind = resistive_forces(10, 80, 0.35, 0.004, 1.225, 0, 0)
        f_headwind = resistive_forces(10, 80, 0.35, 0.004, 1.225, 0, 5)
        self.assertGreater(f_headwind.aero, f_no_wind.aero)


class TestSolveSpeed(unittest.TestCase):
    def setUp(self) -> None:
        self.rider = RiderParams(power_watts=200, weight_kg=75, height_cm=178)
        self.bike = BikeParams()
        self.flat = CourseParams()

    def test_flat_road_reasonable(self) -> None:
        result = solve_speed(self.rider, self.bike, self.flat)
        self.assertGreater(result.speed_kmh, 25)
        self.assertLess(result.speed_kmh, 45)

    def test_climb_slower(self) -> None:
        flat = solve_speed(self.rider, self.bike, self.flat)
        climb = solve_speed(self.rider, self.bike, CourseParams(grade_pct=5))
        self.assertGreater(flat.speed_kmh, climb.speed_kmh)

    def test_descent_faster(self) -> None:
        flat = solve_speed(self.rider, self.bike, self.flat)
        descent = solve_speed(self.rider, self.bike, CourseParams(grade_pct=-5))
        self.assertGreater(descent.speed_kmh, flat.speed_kmh)

    def test_headwind_slower(self) -> None:
        flat = solve_speed(self.rider, self.bike, self.flat)
        wind = solve_speed(self.rider, self.bike, CourseParams(headwind_kmh=20))
        self.assertGreater(flat.speed_kmh, wind.speed_kmh)

    def test_tailwind_via_direction(self) -> None:
        flat = solve_speed(self.rider, self.bike, self.flat)
        tail = solve_speed(self.rider, self.bike,
                           CourseParams(headwind_kmh=20, wind_direction_deg=180))
        self.assertGreater(tail.speed_kmh, flat.speed_kmh)

    def test_more_power_faster(self) -> None:
        low = solve_speed(self.rider, self.bike, self.flat, power_override=100)
        high = solve_speed(self.rider, self.bike, self.flat, power_override=400)
        self.assertGreater(high.speed_kmh, low.speed_kmh)

    def test_zero_power_flat(self) -> None:
        result = solve_speed(
            RiderParams(power_watts=0), self.bike, self.flat
        )
        self.assertAlmostEqual(result.speed_kmh, 0.0)

    def test_coasting_downhill(self) -> None:
        result = solve_speed(
            RiderParams(power_watts=0), self.bike,
            CourseParams(grade_pct=-5),
        )
        self.assertGreater(result.speed_kmh, 30)

    def test_altitude_faster_on_flat(self) -> None:
        sea = solve_speed(self.rider, self.bike, CourseParams(elevation_m=0))
        alt = solve_speed(self.rider, self.bike, CourseParams(elevation_m=2500))
        self.assertGreater(alt.speed_kmh, sea.speed_kmh)

    def test_power_breakdown_sums(self) -> None:
        result = solve_speed(self.rider, self.bike, self.flat)
        total = (abs(result.power_aero) + abs(result.power_rolling)
                 + abs(result.power_gravity) + abs(result.power_drivetrain_loss))
        self.assertAlmostEqual(total, self.rider.power_watts, delta=2.0)

    def test_mph_conversion(self) -> None:
        result = solve_speed(self.rider, self.bike, self.flat)
        self.assertAlmostEqual(
            result.speed_mph, result.speed_kmh * 0.621371, delta=0.1
        )

    def test_aero_position_effect(self) -> None:
        aero_bike = BikeParams(position=RidingPosition.AEROBARS)
        upright_bike = BikeParams(position=RidingPosition.UPRIGHT)
        r_aero = solve_speed(self.rider, aero_bike, self.flat)
        r_upright = solve_speed(self.rider, upright_bike, self.flat)
        self.assertGreater(r_aero.speed_kmh, r_upright.speed_kmh)

    def test_tire_type_effect(self) -> None:
        road_bike = BikeParams(tire_type=TireType.ROAD_RACING)
        mtb_bike = BikeParams(tire_type=TireType.MOUNTAIN)
        r_road = solve_speed(self.rider, road_bike, self.flat)
        r_mtb = solve_speed(self.rider, mtb_bike, self.flat)
        self.assertGreater(r_road.speed_kmh, r_mtb.speed_kmh)


class TestSpeedVsPowerCurve(unittest.TestCase):
    def test_returns_correct_shape(self) -> None:
        rider = RiderParams()
        bike = BikeParams()
        course = CourseParams()
        powers, speeds = speed_vs_power_curve(rider, bike, course, num_points=20)
        self.assertEqual(len(powers), 20)
        self.assertEqual(len(speeds), 20)

    def test_monotonically_increasing(self) -> None:
        rider = RiderParams()
        bike = BikeParams()
        course = CourseParams()
        powers, speeds = speed_vs_power_curve(rider, bike, course)
        for i in range(1, len(speeds)):
            self.assertGreaterEqual(speeds[i], speeds[i - 1])


# -----------------------------------------------------------------------
# New feature tests
# -----------------------------------------------------------------------

class TestWindDirection(unittest.TestCase):
    def test_headwind_is_full_speed(self) -> None:
        hw = effective_headwind(20, 0)
        self.assertAlmostEqual(hw, 20.0, places=5)

    def test_tailwind_is_negative(self) -> None:
        hw = effective_headwind(20, 180)
        self.assertAlmostEqual(hw, -20.0, places=5)

    def test_crosswind_is_zero(self) -> None:
        hw = effective_headwind(20, 90)
        self.assertAlmostEqual(hw, 0.0, places=5)

    def test_45_degrees(self) -> None:
        hw = effective_headwind(20, 45)
        expected = 20 * math.cos(math.radians(45))
        self.assertAlmostEqual(hw, expected, places=5)


class TestWheelInertia(unittest.TestCase):
    def test_factor_greater_than_one(self) -> None:
        factor = wheel_inertia_factor(1.8, 0.34, 83)
        self.assertGreater(factor, 1.0)

    def test_heavier_wheels_higher_factor(self) -> None:
        light = wheel_inertia_factor(1.0, 0.34, 83)
        heavy = wheel_inertia_factor(3.0, 0.34, 83)
        self.assertGreater(heavy, light)


class TestValidation(unittest.TestCase):
    def test_valid_params_no_errors(self) -> None:
        errors = validate_all(RiderParams(), BikeParams(), CourseParams())
        self.assertEqual(errors, [])

    def test_power_too_high(self) -> None:
        err = validate_param("power_watts", 5000)
        self.assertIsNotNone(err)

    def test_power_in_range(self) -> None:
        err = validate_param("power_watts", 200)
        self.assertIsNone(err)

    def test_negative_weight(self) -> None:
        err = validate_param("rider_weight_kg", -5)
        self.assertIsNotNone(err)

    def test_extreme_grade(self) -> None:
        err = validate_param("grade_pct", 50)
        self.assertIsNotNone(err)


class TestUnitConversions(unittest.TestCase):
    def test_kg_lbs_roundtrip(self) -> None:
        self.assertAlmostEqual(lbs_to_kg(kg_to_lbs(75)), 75, places=3)

    def test_cm_inches_roundtrip(self) -> None:
        self.assertAlmostEqual(inches_to_cm(cm_to_inches(178)), 178, places=3)

    def test_m_feet_roundtrip(self) -> None:
        self.assertAlmostEqual(feet_to_m(m_to_feet(100)), 100, places=3)

    def test_celsius_fahrenheit_roundtrip(self) -> None:
        self.assertAlmostEqual(
            fahrenheit_to_celsius(celsius_to_fahrenheit(20)), 20, places=3)

    def test_kmh_mph_roundtrip(self) -> None:
        self.assertAlmostEqual(mph_to_kmh(kmh_to_mph(30)), 30, places=3)

    def test_freezing_point(self) -> None:
        self.assertAlmostEqual(celsius_to_fahrenheit(0), 32, places=3)

    def test_boiling_point(self) -> None:
        self.assertAlmostEqual(celsius_to_fahrenheit(100), 212, places=3)


class TestCourseProfile(unittest.TestCase):
    def test_single_flat_segment(self) -> None:
        rider = RiderParams(power_watts=200, weight_kg=75, height_cm=178)
        bike = BikeParams()
        segments = [CourseSegment(distance_m=5000, grade_pct=0)]
        result = simulate_course_profile(rider, bike, segments)
        self.assertEqual(len(result.segments), 1)
        self.assertAlmostEqual(result.total_distance_m, 5000)
        self.assertGreater(result.avg_speed_kmh, 0)

    def test_multi_segment(self) -> None:
        rider = RiderParams(power_watts=200, weight_kg=75, height_cm=178)
        bike = BikeParams()
        segments = [
            CourseSegment(distance_m=2000, grade_pct=0),
            CourseSegment(distance_m=1000, grade_pct=8),
            CourseSegment(distance_m=1000, grade_pct=-5),
        ]
        result = simulate_course_profile(rider, bike, segments)
        self.assertEqual(len(result.segments), 3)
        self.assertAlmostEqual(result.total_distance_m, 4000)
        self.assertGreater(result.total_elevation_gain_m, 0)
        self.assertGreater(result.total_elevation_loss_m, 0)

    def test_uphill_slower_than_flat(self) -> None:
        rider = RiderParams(power_watts=200, weight_kg=75, height_cm=178)
        bike = BikeParams()
        segments = [
            CourseSegment(distance_m=1000, grade_pct=0),
            CourseSegment(distance_m=1000, grade_pct=10),
        ]
        result = simulate_course_profile(rider, bike, segments)
        self.assertGreater(result.segments[0].speed_kmh,
                           result.segments[1].speed_kmh)

    def test_elevation_tracking(self) -> None:
        rider = RiderParams(power_watts=200, weight_kg=75, height_cm=178)
        bike = BikeParams()
        segments = [
            CourseSegment(distance_m=1000, grade_pct=5, elevation_m=100),
        ]
        result = simulate_course_profile(rider, bike, segments)
        self.assertGreater(result.elevations[-1], result.elevations[0])


class TestPresetSaveLoad(unittest.TestCase):
    def test_roundtrip(self) -> None:
        presets = [
            Preset(
                name="Test",
                rider=RiderParams(power_watts=250, weight_kg=70, height_cm=175),
                bike=BikeParams(weight_kg=7, tire_type=TireType.ROAD_RACING,
                                position=RidingPosition.DROPS),
                course=CourseParams(grade_pct=3, headwind_kmh=10, elevation_m=500),
            )
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        save_presets(presets, path)
        loaded = load_presets(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].name, "Test")
        self.assertAlmostEqual(loaded[0].rider.power_watts, 250)
        self.assertEqual(loaded[0].bike.tire_type, TireType.ROAD_RACING)
        Path(path).unlink()


class TestExportCSV(unittest.TestCase):
    def test_export_result(self) -> None:
        rider = RiderParams(power_watts=200)
        bike = BikeParams()
        course = CourseParams()
        result = solve_speed(rider, bike, course)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        export_result_csv(result, rider, bike, course, path)
        content = Path(path).read_text()
        self.assertIn("Speed (km/h)", content)
        self.assertIn("200", content)
        Path(path).unlink()

    def test_export_course(self) -> None:
        rider = RiderParams(power_watts=200)
        bike = BikeParams()
        segments = [CourseSegment(distance_m=1000)]
        result = simulate_course_profile(rider, bike, segments)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        export_course_csv(result, path)
        content = Path(path).read_text()
        self.assertIn("Segment", content)
        Path(path).unlink()

    def test_export_comparison(self) -> None:
        rider = RiderParams(power_watts=200)
        bike = BikeParams()
        course = CourseParams()
        result = solve_speed(rider, bike, course)
        scenarios = [("Test", rider, bike, course, result)]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        export_comparison_csv(scenarios, path)
        content = Path(path).read_text()
        self.assertIn("Scenario", content)
        Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
