import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ac_dataset.tcp.data import preprocess_rgb
from ac_dataset.tcp.shadow import (
    RacingLineTarget, discover_racing_line, observed_controls,
    live_ego_features, standing_lap_from_state,
)


class TCPShadowTests(unittest.TestCase):
    def test_observed_controls_prefers_normalized_steer(self):
        result = observed_controls({"steerNorm": -.2, "steerAngle": 30,
                                    "accStatus": .7, "brakeStatus": .1})
        self.assertEqual(result["actual_steer"], -.2)
        self.assertEqual(result["actual_steer_angle_deg"], 30)
        self.assertAlmostEqual(result["actual_signed_accel"], .6)

    def test_live_lap_count_maps_to_training_phase(self):
        self.assertTrue(standing_lap_from_state({"LapCount": 0}))
        self.assertFalse(standing_lap_from_state({"LapCount": 1}))

    def test_live_ego_uses_local_signals_and_converts_g(self):
        raw, normalized = live_ego_features({
            "speed": 45, "local_velocity_x": 44, "local_velocity_y": -2,
            "angular_velocity_y": 1, "accelY": 2, "accelX": -1,
        })
        np.testing.assert_allclose(
            raw, [45, 44, -2, 1, 19.6133, -9.80665], rtol=1e-5)
        np.testing.assert_allclose(
            normalized, [0.5, 44 / 90, -.2, .5, 19.6133 / 50, -9.80665 / 50],
            rtol=1e-5)

    def test_racing_line_target_matches_prepare_search_rule(self):
        line = pd.DataFrame({
            "normalized_position": [0, .25, .5, .75],
            "x": [0, 25, 50, 75], "y": [0, 0, 0, 0],
            "dist": [0, 25, 50, 75],
        })
        provider = RacingLineTarget(line)
        np.testing.assert_array_equal(provider.target_world_points(0)[6], [50, 0])
        np.testing.assert_array_equal(provider.target_world(0), [0, 0])
        self.assertAlmostEqual(provider.deviation_m(.25, [25, 3]), 3.0)

    def test_physical_deviation_uses_nearest_segment_not_spline_index(self):
        line = pd.DataFrame({
            "normalized_position": [0, .1, .9],
            "x": [0, 10, 20], "y": [0, 0, 0],
            "dist": [0, 10, 20],
        })
        provider = RacingLineTarget(line)
        self.assertGreater(provider.indexed_deviation_m(.9, [5, 2]), 10)
        self.assertAlmostEqual(provider.deviation_m(.9, [5, 2]), 2.0)

    def test_control_line_starts_at_nearest_projection_and_wraps(self):
        line = pd.DataFrame({
            "normalized_position": [0, .25, .5, .75],
            "x": [0, 10, 10, 0], "y": [0, 0, 10, 10],
            "dist": [0, 10, 20, 30],
        })
        provider = RacingLineTarget(line)
        local, distance, deviation = provider.control_local_points([1, 10], 0)
        self.assertAlmostEqual(distance, 29.0)
        self.assertAlmostEqual(deviation, 0.0)
        np.testing.assert_allclose(local[0], [0, 0], atol=1e-6)
        self.assertTrue(np.isfinite(local).all())

    def test_rgb_preprocessing_shape(self):
        image = np.zeros((945, 1920, 3), dtype=np.uint8)
        tensor = preprocess_rgb(image)
        self.assertEqual(tuple(tensor.shape), (3, 256, 512))

    def test_racing_line_discovery_skips_legacy_session_without_track_info(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            prepared = root / "prepared"
            prepared.mkdir()
            first = root / "solo" / "old" / "lap_01" / "CAMERA" / "0.jpg"
            second = root / "solo" / "new" / "lap_01" / "CAMERA" / "0.jpg"
            line = second.parents[2] / "TRACK_INFO" / "AI_RACING_LINE.csv"
            line.parent.mkdir(parents=True)
            line.write_text("normalized_position,x,y,dist\n0,0,0,0\n", encoding="utf-8")
            pd.DataFrame({"session": ["old", "new"],
                          "image_path": [str(first), str(second)]}).to_parquet(
                prepared / "manifest.parquet", index=False)
            self.assertEqual(discover_racing_line(prepared), line)


if __name__ == "__main__":
    unittest.main()
