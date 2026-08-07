import unittest
from pathlib import Path
import tempfile

import numpy as np

from ac_dataset.control.vjoy import VJoyDevice
from ac_dataset.tcp.closed_loop import (
    FrameWriter, KeyboardController, StallDetector, limited_controls,
)


class VJoyMappingTests(unittest.TestCase):
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

    def test_initial_closed_loop_limits(self):
        pred = {"pred_steer": .8, "pred_throttle": .9, "pred_brake": .7}
        positive = limited_controls(pred, 1, .2, .4, .3, 1)
        negative = limited_controls(pred, -1, .2, .4, .3, .5)
        self.assertAlmostEqual(positive[0], .08)
        self.assertEqual(positive[1:], (.3, .7))
        self.assertAlmostEqual(negative[0], -.08)
        self.assertEqual(negative[1:], (.3, .5))

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
