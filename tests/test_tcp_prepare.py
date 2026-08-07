import unittest

import numpy as np

from ac_dataset.tcp.data import quaternion_yaw, to_local
from ac_dataset.tcp.config import (
    POSITION_SCALE_M, SPEED_SCALE_MPS, lap_phase_metadata, normalization_metadata,
)
from ac_dataset.tcp.data import route_command
from ac_dataset.tcp.prepare import parse_ego_info


class TCPPrepareTests(unittest.TestCase):
    def test_tcp_physical_scales_are_checkpoint_serializable(self):
        self.assertEqual(POSITION_SCALE_M, 50.0)
        self.assertEqual(SPEED_SCALE_MPS, 90.0)
        self.assertEqual(normalization_metadata(), {
            "position_scale_m": 50.0,
            "speed_scale_mps": 90.0,
        })

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
                "throttle: 0.7\nbrake: 0.1\nsteer_angle: -0.2\n", encoding="utf-8")
            row = parse_ego_info(path)
        self.assertEqual(row["speed_mps"], 5.0)
        self.assertEqual(row["steer"], -0.2)

    def test_local_coordinates_face_positive_y(self):
        yaw = quaternion_yaw(0.0, 0.0, 0.0, 1.0)
        local = to_local(np.array([[10.0, 0.0], [0.0, 2.0]]), np.array([0.0, 0.0]), yaw)
        np.testing.assert_allclose(local, [[0.0, 10.0], [2.0, 0.0]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
