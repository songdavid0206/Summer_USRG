import unittest

import numpy as np

from ac_dataset.processed_builder import _detect_retirement


class RetirementDetectionTests(unittest.TestCase):
    def test_multistage_pit_move_uses_first_stable_teleport(self):
        x = np.r_[0.0, 1.0, 2.0, np.full(30, 100.0), np.full(30, 120.0)]
        y = np.zeros_like(x)
        speed = np.r_[80.0, 80.0, 80.0, np.zeros(60)]
        self.assertEqual(_detect_retirement(x, y, speed), 3)

    def test_short_teleport_glitch_that_resumes_is_not_retirement(self):
        x = np.r_[0.0, 1.0, 2.0, np.full(10, 100.0), np.arange(101.0, 131.0)]
        y = np.zeros_like(x)
        speed = np.r_[80.0, 80.0, 80.0, np.zeros(10), np.full(30, 80.0)]
        self.assertIsNone(_detect_retirement(x, y, speed))

    def test_track_stop_without_teleport_is_not_retirement(self):
        x = np.r_[np.arange(20.0), np.full(40, 19.0)]
        y = np.zeros_like(x)
        speed = np.r_[np.full(20, 80.0), np.zeros(40)]
        self.assertIsNone(_detect_retirement(x, y, speed))


if __name__ == "__main__":
    unittest.main()
