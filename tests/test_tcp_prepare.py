import unittest

import numpy as np

from ac_dataset.tcp.data import quaternion_yaw, to_local
from ac_dataset.tcp.config import (
    EGO_FEATURE_NAMES, POSITION_SCALE_M, RACING_LINE_LOOKAHEADS_M,
    RACING_LINE_SCALE_M,
    SPEED_SCALE_MPS, lap_phase_metadata, normalization_metadata,
)
from ac_dataset.tcp.data import normalize_ego_features, route_command
from ac_dataset.tcp.prepare import parse_ego_info


class TCPPrepareTests(unittest.TestCase):
    def test_tcp_physical_scales_are_checkpoint_serializable(self):
        self.assertEqual(POSITION_SCALE_M, 50.0)
        self.assertEqual(SPEED_SCALE_MPS, 90.0)
        metadata = normalization_metadata()
        self.assertEqual(metadata["schema_version"], 3)
        self.assertEqual(metadata["position_scale_m"], 50.0)
        self.assertEqual(metadata["speed_scale_mps"], 90.0)
        self.assertEqual(RACING_LINE_SCALE_M, 150.0)
        self.assertEqual(metadata["racing_line_scale_m"], 150.0)
        self.assertEqual(tuple(metadata["ego_features"]), EGO_FEATURE_NAMES)
        self.assertEqual(tuple(metadata["racing_line_lookaheads_m"]),
                         RACING_LINE_LOOKAHEADS_M)

    def test_ego_normalization_has_no_absolute_pose(self):
        raw = np.array([90.0, 45.0, 5.0, 1.0, 25.0, 25.0])
        np.testing.assert_allclose(
            normalize_ego_features(raw), [1.0, .5, .5, .5, .5, .5])
        self.assertFalse(any(name in EGO_FEATURE_NAMES for name in ("pos_x", "pos_y", "yaw")))

    def test_standing_and_flying_laps_use_distinct_command_slots(self):
        np.testing.assert_array_equal(route_command("standing"), [0, 0, 0, 1, 1, 0])
        np.testing.assert_array_equal(route_command("flying"), [0, 0, 0, 1, 0, 1])
        self.assertEqual(lap_phase_metadata()["standing_dataset_lap"], 1)

    def test_parse_ego_info(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ego.txt"
            path.write_text(
                "position: 1 2 3\norientation: 0 0 0 1\nvelocity: 3 4 0\n"
                "angular_velocity: 1 2 90\nacceleration: 5 6 0\n"
                "throttle: 0.7\nbrake: 0.1\nsteer_angle: -0.2\n", encoding="utf-8")
            row = parse_ego_info(path)
        self.assertEqual(row["speed_mps"], 5.0)
        self.assertEqual(row["steer"], -0.2)
        self.assertAlmostEqual(row["yaw_rate_rad_s"], np.pi / 2)
        self.assertEqual(row["longitudinal_accel_mps2"], 5.0)

    def test_local_coordinates_face_positive_y(self):
        yaw = quaternion_yaw(0.0, 0.0, 0.0, 1.0)
        local = to_local(np.array([[10.0, 0.0], [0.0, 2.0]]), np.array([0.0, 0.0]), yaw)
        np.testing.assert_allclose(local, [[0.0, 10.0], [2.0, 0.0]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
