import unittest
from pathlib import Path
import tempfile

import numpy as np

from ac_dataset.control.vjoy import VJoyDevice
from ac_dataset.control.steer_map import SteerMap
from ac_dataset.control.latency_probe import summarize
from ac_dataset.tcp.closed_loop import (
    ExpertSpeedProfile, FrameWriter, KeyboardController, SpeedProfileGuard,
    PreviewPIDLongitudinalController, StallDetector,
    WaypointPurePursuitController, control_lap_phases, limited_controls,
    slip_limited_throttle,
)


class VJoyMappingTests(unittest.TestCase):
    def test_latency_summary_reports_each_segment(self):
        rows = [{
            "client_to_bridge_receive_ms": value,
            "bridge_update_vjd_ms": value + 1,
            "client_roundtrip_ms": value + 2,
            "bridge_apply_to_ac_telemetry_ms": value + 3,
            "client_send_to_ac_telemetry_ms": value + 4,
        } for value in (1.0, 2.0, 3.0)]
        result = summarize(rows)
        self.assertEqual(result["transitions"], 3)
        self.assertEqual(result["client_to_bridge_receive"]["median_ms"], 2.0)
        self.assertEqual(result["bridge_update_vjd"]["median_ms"], 3.0)
        self.assertEqual(result["client_send_to_ac_telemetry"]["median_ms"], 6.0)

    def test_axis_mapping(self):
        self.assertEqual(VJoyDevice._axis(-1, centered=True), 0)
        self.assertEqual(VJoyDevice._axis(0, centered=True), 0x4000)
        self.assertEqual(VJoyDevice._axis(1, centered=True), 0x8000)
        self.assertEqual(VJoyDevice._axis(0), 0)
        self.assertEqual(VJoyDevice._axis(1), 0x8000)

    def test_axis_mapping_clamps(self):
        self.assertEqual(VJoyDevice._axis(-2, centered=True), 0)
        self.assertEqual(VJoyDevice._axis(2, centered=True), 0x8000)
        self.assertEqual(VJoyDevice._axis(-1), 0)
        self.assertEqual(VJoyDevice._axis(2), 0x8000)

    def test_steer_map_interpolates_normalized_physical_angle(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "steer_map.csv"
            path.write_text(
                "action_steerAngle,steerAngle\n-0.72,-180\n0.72,180\n",
                encoding="utf-8")
            mapping = SteerMap.from_csv(path)
            self.assertAlmostEqual(mapping.action_for_normalized(.5), .36)
            self.assertAlmostEqual(mapping.action_for_normalized(-1), -.72)

    def test_initial_closed_loop_limits(self):
        pred = {"pred_steer": .8, "pred_throttle": .9, "pred_brake": .7}
        positive = limited_controls(pred, 1, .2, .4, .3, 1)
        negative = limited_controls(pred, -1, .2, .4, .3, .5)
        self.assertAlmostEqual(positive[0], .08)
        self.assertEqual(positive[1:], (.3, .7))
        self.assertAlmostEqual(negative[0], -.08)
        self.assertEqual(negative[1:], (.3, .5))

    def test_expert_speed_profile_interpolates_and_wraps_by_phase(self):
        profile = ExpertSpeedProfile({
            "standing": np.array([10.0, 20.0, 30.0, 40.0]),
            "flying": np.array([50.0, 60.0, 70.0, 80.0]),
        }, .75, {"standing": 4, "flying": 4})
        self.assertAlmostEqual(profile.target_mps(.125, True), 10.0)
        self.assertAlmostEqual(profile.target_mps(.375, False), 60.0)
        self.assertAlmostEqual(profile.target_mps(0.0, True), 25.0)
        self.assertAlmostEqual(profile.target_mps(1.0, True), 25.0)

    def test_speed_profile_guard_is_inactive_inside_margin(self):
        profile = ExpertSpeedProfile({
            "standing": np.full(4, 50.0), "flying": np.full(4, 60.0),
        }, .75, {"standing": 4, "flying": 4})
        guard = SpeedProfileGuard(profile, margin_mps=2.0)
        throttle, brake, state = guard.apply(51.5, .2, True, .8, .1, 1.0)
        self.assertAlmostEqual(throttle, .8)
        self.assertAlmostEqual(brake, .1)
        self.assertFalse(state["speed_guard_active"])

    def test_speed_profile_guard_cuts_power_and_adds_braking(self):
        profile = ExpertSpeedProfile({
            "standing": np.full(4, 50.0), "flying": np.full(4, 60.0),
        }, .75, {"standing": 4, "flying": 4})
        guard = SpeedProfileGuard(
            profile, margin_mps=1.0,
            throttle_cut_range_mps=3.0, full_brake_range_mps=8.0)
        throttle, brake, state = guard.apply(55.0, .2, True, 1.0, .1, 1.0)
        self.assertEqual(throttle, 0.0)
        self.assertAlmostEqual(brake, .5)
        self.assertTrue(state["speed_guard_active"])
        self.assertAlmostEqual(state["speed_profile_overspeed_mps"], 5.0)

    def test_speed_profile_guard_respects_brake_limit(self):
        profile = ExpertSpeedProfile({
            "standing": np.full(4, 50.0), "flying": np.full(4, 50.0),
        }, .75, {"standing": 4, "flying": 4})
        guard = SpeedProfileGuard(profile, margin_mps=0.0)
        throttle, brake, _ = guard.apply(70.0, .2, True, 1.0, 0.0, .4)
        self.assertEqual(throttle, 0.0)
        self.assertAlmostEqual(brake, .4)

    @staticmethod
    def _straight_racing_line(length=1000.0):
        class Line:
            dist = np.array([0.0, length])
            norm = np.array([0.0, 1.0])

        line = Line()
        line.length = length
        return line

    def test_preview_pid_accelerates_and_brakes_with_exclusive_pedals(self):
        profile = ExpertSpeedProfile({
            "standing": np.full(100, 20.0), "flying": np.full(100, 20.0),
        }, .5, {"standing": 100, "flying": 100})
        controller = PreviewPIDLongitudinalController(
            profile, self._straight_racing_line(), kp=.2, ki=0, kd=0)
        throttle, brake, state = controller.update(10, .2, True, .025)
        self.assertGreater(throttle, 0)
        self.assertEqual(brake, 0)
        self.assertEqual(state["longitudinal_controller"], "profile_pid")
        throttle, brake, _ = controller.update(30, .2, True, .025)
        self.assertEqual(throttle, 0)
        self.assertGreater(brake, 0)

    def test_preview_pid_brakes_before_a_future_speed_drop(self):
        class StepProfile:
            quantile = .5
            bins = 100
            sample_counts = {"standing": 100, "flying": 100}

            @staticmethod
            def target_mps(spline_position, standing_lap):
                return 50.0 if spline_position < .11 else 10.0

            def metadata(self):
                return {"kind": "step"}

        controller = PreviewPIDLongitudinalController(
            StepProfile(), self._straight_racing_line(), lookaheads_m=(0, 20),
            preview_decel_mps2=14, kp=.2, ki=0, kd=0)
        throttle, brake, state = controller.update(30, .1, True, .025)
        self.assertEqual(throttle, 0)
        self.assertGreater(brake, 0)
        self.assertEqual(state["longitudinal_limiting_lookahead_m"], 20)
        self.assertLess(state["longitudinal_target_mps"], 30)

    def test_preview_pid_target_scale_adds_speed_margin(self):
        profile = ExpertSpeedProfile({
            "standing": np.full(100, 20.0), "flying": np.full(100, 20.0),
        }, .5, {"standing": 100, "flying": 100})
        controller = PreviewPIDLongitudinalController(
            profile, self._straight_racing_line(), lookaheads_m=(0,),
            kp=.2, ki=0, kd=0, target_scale=.9)
        _, brake, state = controller.update(19, .2, True, .025)
        self.assertGreater(brake, 0)
        self.assertAlmostEqual(state["longitudinal_profile_now_mps"], 20)
        self.assertAlmostEqual(state["longitudinal_scaled_profile_now_mps"], 18)

    def test_pure_pursuit_follows_learned_waypoint_direction(self):
        left = WaypointPurePursuitController(steer_rate_per_s=100)
        right = WaypointPurePursuitController(steer_rate_per_s=100)
        straight = WaypointPurePursuitController(steer_rate_per_s=100)
        left_steer, _ = left.update(
            np.array([[-1., 5.], [-3., 12.]]), 20, .025)
        right_steer, _ = right.update(
            np.array([[1., 5.], [3., 12.]]), 20, .025)
        straight_steer, _ = straight.update(
            np.array([[0., 5.], [0., 12.]]), 20, .025)
        self.assertGreater(left_steer, 0)
        self.assertLess(right_steer, 0)
        self.assertAlmostEqual(straight_steer, 0)

    def test_pure_pursuit_rate_limits_steering(self):
        controller = WaypointPurePursuitController(steer_rate_per_s=2.0)
        steer, state = controller.update(
            np.array([[-2., 5.], [-5., 10.]]), 20, .025)
        self.assertAlmostEqual(steer, .05)
        self.assertGreater(state["lateral_limited_steer"], steer)

    def test_pure_pursuit_cross_track_term_returns_to_racing_line(self):
        controller = WaypointPurePursuitController(
            steer_rate_per_s=100, cross_track_gain=1.5)
        steer, state = controller.update(
            np.array([[0., 5.], [0., 15.]]), 20, .025,
            racing_line_local_m=np.array([[3., 0.], [3., 10.]]))
        self.assertLess(steer, 0)
        self.assertLess(state["lateral_cross_track_correction"], 0)
        self.assertAlmostEqual(state["lateral_pursuit_steer"], 0)

    def test_curvature_adaptive_lookahead_shortens_tight_path(self):
        path = np.array([[-1., 5.], [-5., 10.], [-10., 15.]])
        flat = WaypointPurePursuitController(
            steer_rate_per_s=100, lookahead_curvature_gain=0)
        adaptive = WaypointPurePursuitController(
            steer_rate_per_s=100, lookahead_curvature_gain=30,
            lookahead_curvature_min_scale=.5)
        _, flat_state = flat.update(path, 30, .025)
        _, adaptive_state = adaptive.update(path, 30, .025)
        self.assertLess(adaptive_state["lateral_desired_lookahead_m"],
                        flat_state["lateral_desired_lookahead_m"])
        self.assertLess(adaptive_state["lateral_curvature_scale"], 1)

    def test_cross_track_gain_is_reduced_at_high_speed(self):
        kwargs = dict(steer_rate_per_s=100, cross_track_gain=1.0,
                      cross_track_highspeed_gain=.2,
                      cross_track_lowspeed_mps=10,
                      cross_track_highspeed_mps=30)
        slow = WaypointPurePursuitController(**kwargs)
        fast = WaypointPurePursuitController(**kwargs)
        path = np.array([[0., 5.], [0., 15.]])
        line = np.array([[3., 0.], [3., 15.]])
        _, slow_state = slow.update(path, 10, .025, racing_line_local_m=line)
        _, fast_state = fast.update(path, 30, .025, racing_line_local_m=line)
        self.assertLess(abs(fast_state["lateral_cross_track_correction"]),
                        abs(slow_state["lateral_cross_track_correction"]))

    def test_model_and_speed_profile_lap_phases_are_independent(self):
        self.assertEqual(control_lap_phases(True, False), (True, True))
        self.assertEqual(control_lap_phases(True, True), (True, False))
        self.assertEqual(control_lap_phases(False, True), (False, False))

    def test_slip_throttle_cut_is_linear_and_reaches_zero(self):
        unchanged, state = slip_limited_throttle(1.0, 8, 10, 20)
        self.assertEqual(unchanged, 1.0)
        self.assertEqual(state, {})
        halfway, state = slip_limited_throttle(1.0, 15, 10, 20)
        self.assertAlmostEqual(halfway, .5)
        self.assertTrue(state["slip_throttle_cut_active"])
        stopped, _ = slip_limited_throttle(1.0, 25, 10, 20)
        self.assertEqual(stopped, 0.0)

    def test_recovery_uses_racing_line_with_hysteresis(self):
        controller = WaypointPurePursuitController(
            steer_rate_per_s=100, recovery_enabled=True,
            recovery_stable_s=.2)
        learned = np.array([[0., 5.], [0., 15.]])
        line = np.array([[3., 0.], [5., 15.]])
        _, active = controller.update(
            learned, 20, .1, racing_line_local_m=line,
            racing_line_deviation_m=4, longitudinal_speed_mps=20)
        self.assertTrue(active["recovery_active"])
        self.assertEqual(active["lateral_source"], "racing_line_recovery")
        controller.update(
            learned, 20, .1, racing_line_local_m=line,
            racing_line_deviation_m=1, longitudinal_speed_mps=20)
        _, recovered = controller.update(
            learned, 20, .1, racing_line_local_m=line,
            racing_line_deviation_m=1, longitudinal_speed_mps=20)
        self.assertFalse(recovered["recovery_active"])
        self.assertEqual(recovered["lateral_source"], "learned_waypoints")

    def test_slip_recovery_cuts_power_and_caps_speed(self):
        controller = WaypointPurePursuitController(
            steer_rate_per_s=100, recovery_enabled=True,
            recovery_slip_angle_rad=.1)
        controller.update(
            np.array([[0., 5.], [0., 15.]]), 25, .025,
            racing_line_local_m=np.array([[0., 0.], [0., 15.]]),
            longitudinal_speed_mps=20, lateral_speed_mps=3)
        throttle, brake, state = controller.apply_recovery_longitudinal(
            50, 60, 1, .5)
        self.assertEqual(throttle, 0)
        self.assertLessEqual(brake, .15)
        self.assertTrue(state["recovery_speed_control_active"])
        self.assertEqual(state["recovery_speed_target_mps"], 45)

    def test_deviation_recovery_uses_moderate_braking(self):
        controller = WaypointPurePursuitController(
            steer_rate_per_s=100, recovery_enabled=True)
        controller.update(
            np.array([[0., 5.], [0., 15.]]), 50, .025,
            racing_line_local_m=np.array([[2., 0.], [3., 15.]]),
            racing_line_deviation_m=4, longitudinal_speed_mps=50)
        throttle, brake, _ = controller.apply_recovery_longitudinal(
            50, 50, 1, 0)
        self.assertEqual(throttle, 0)
        self.assertGreater(brake, 0)
        self.assertLessEqual(brake, .25)

    def test_stall_detector_ignores_initial_grid_stop(self):
        detector = StallDetector(10, 1, 1)
        self.assertFalse(detector.update(0, 0))
        self.assertFalse(detector.update(0, 5))

    def test_stall_detector_requires_persistent_stop_after_moving(self):
        detector = StallDetector(10, 1, 1)
        self.assertFalse(detector.update(12, 0))
        self.assertFalse(detector.update(.5, 1))
        self.assertFalse(detector.update(.5, 1.9))
        self.assertTrue(detector.update(.5, 2.0))
        self.assertFalse(detector.update(2, 2.1))

    def test_frame_writer_saves_resized_jpeg(self):
        with tempfile.TemporaryDirectory() as raw:
            writer = FrameWriter(Path(raw), width=100)
            relative = writer.submit(40, 1.0, np.zeros((50, 200, 3), np.uint8))
            writer.close()
            self.assertEqual(relative, "frames/000040_001.000s.jpg")
            self.assertEqual(writer.saved, 1)
            self.assertEqual(writer.dropped, 0)
            self.assertEqual(writer.errors, [])
            self.assertTrue((Path(raw) / "000040_001.000s.jpg").is_file())

    def test_keyboard_controller_is_speed_sensitive(self):
        slow = KeyboardController(2, 3.5, 4)
        fast = KeyboardController(2, 3.5, 4)
        self.assertEqual(slow.update(False, True, False, False, 0, 1)[0], 1.0)
        self.assertAlmostEqual(fast.update(False, True, False, False, 80, 1)[0], .22)

    def test_keyboard_brake_overrides_throttle(self):
        controller = KeyboardController(2, 3.5, 4)
        _, throttle, brake = controller.update(False, False, True, True, 0, .25)
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 1.0)


if __name__ == "__main__":
    unittest.main()
