import unittest

import numpy as np
import pandas as pd

from ac_dataset.recorder.dataset_builder import _pair_by_gametime


class LapPairingTest(unittest.TestCase):
    def test_lap_reset_outlier_does_not_mix_laps(self):
        telemetry = pd.DataFrame({
            "t": [0.00, 0.01, 1.00, 1.01, 1.02, 2.00],
            "timestamp_ac": [10.00, 10.01, 11.00, 11.01, 11.02, 12.00],
            "currentTime": [900, 910, 990, 3, 10, 980],
            # At 1.01 s currentTime reset one tick before completedLaps.
            "completedLaps": [0, 0, 0, 0, 1, 1],
            "LapCount": [0, 0, 1, 1, 1, 1],
            "NormalizedSplinePosition": [.90, .91, .999, .001, .01, .99],
            "speed": [10, 10, 10, 20, 20, 20],
        })
        images = pd.DataFrame({
            "frame": [0, 1],
            "t": [1.03, 1.99],
            "game_time_ms": [20, 970],
            "completed_laps": [1, 1],
            "image_path": ["camera/000000.jpg", "camera/000001.jpg"],
        })
        paired = _pair_by_gametime(images, telemetry, tolerance=.05)
        self.assertEqual(paired["LapCount"].tolist(), [1.0, 1.0])
        self.assertLess(float(paired["NormalizedSplinePosition"].iloc[0]), .05)
        self.assertGreater(float(paired["NormalizedSplinePosition"].iloc[1]), .9)
        self.assertTrue(np.allclose(paired["speed"], 20))

    def test_reset_before_lap_counter_uses_wall_time_only_to_choose_cycle(self):
        telemetry = pd.DataFrame({
            "t": [0.00, 0.99, 1.00, 1.01, 1.99, 2.00, 2.01],
            "timestamp_ac": [10.00, 10.99, 11.00, 11.01, 11.99, 12.00, 12.01],
            "currentTime": [0, 990, 3, 10, 990, 3, 10],
            "completedLaps": [0, 0, 1, 1, 1, 2, 2],
            "LapCount": [0, 1, 1, 1, 2, 2, 2],
            "NormalizedSplinePosition": [0, .999, .001, .01, .999, .001, .01],
            "speed": [10, 10, 20, 20, 20, 30, 30],
        })
        # Graphics lap counter lags one tick: at the second reset it still says
        # lap 1, time 3 ms. It must map near wall time 2.00, not lap 1 start.
        images = pd.DataFrame({
            "frame": [0],
            "t": [2.001],
            "game_time_ms": [3],
            "completed_laps": [1],
            "image_path": ["camera/000000.jpg"],
        })
        paired = _pair_by_gametime(images, telemetry, tolerance=.05)
        self.assertGreater(float(paired["speed"].iloc[0]), 25)
        self.assertEqual(float(paired["LapCount"].iloc[0]), 2.0)


if __name__ == "__main__":
    unittest.main()
