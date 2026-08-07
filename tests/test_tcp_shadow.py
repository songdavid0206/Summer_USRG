import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ac_dataset.tcp.data import preprocess_rgb
from ac_dataset.tcp.shadow import (
    RacingLineTarget, decode_signed_accel, discover_racing_line, observed_controls,
    standing_lap_from_state,
)


class TCPShadowTests(unittest.TestCase):
    def test_signed_accel_decodes_to_one_pedal(self):
        self.assertEqual(decode_signed_accel(.4), (.4, 0.0))
        self.assertEqual(decode_signed_accel(-.3), (0.0, .3))
        self.assertEqual(decode_signed_accel(0), (0.0, 0.0))

    def test_observed_controls_prefers_normalized_steer(self):
        result = observed_controls({"steerNorm": -.2, "steerAngle": 30,
                                    "accStatus": .7, "brakeStatus": .1})
        self.assertEqual(result["actual_steer"], -.2)
        self.assertAlmostEqual(result["actual_signed_accel"], .6)

    def test_live_lap_count_maps_to_training_phase(self):
        self.assertTrue(standing_lap_from_state({"LapCount": 0}))
        self.assertFalse(standing_lap_from_state({"LapCount": 1}))

    def test_racing_line_target_matches_prepare_search_rule(self):
        line = pd.DataFrame({
            "normalized_position": [0, .25, .5, .75],
            "x": [0, 25, 50, 75], "y": [0, 0, 0, 0],
            "dist": [0, 25, 50, 75],
        })
        provider = RacingLineTarget(line, lookahead_m=50)
        np.testing.assert_array_equal(provider.target_world(0), [50, 0])

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
