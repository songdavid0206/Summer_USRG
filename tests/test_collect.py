import unittest

from ac_dataset.collect import _lap_crossing_goal, _retained_complete_laps


class LapTargetTest(unittest.TestCase):
    def test_starting_just_after_line_still_discards_unproven_fragment(self):
        self.assertEqual(_lap_crossing_goal(5, 0.02), 6)

    def test_starting_before_line_needs_extra_crossing(self):
        self.assertEqual(_lap_crossing_goal(5, 0.985), 6)

    def test_starting_mid_lap_needs_extra_crossing(self):
        self.assertEqual(_lap_crossing_goal(5, 0.5), 6)

    def test_progress_count_excludes_first_observed_fragment(self):
        self.assertEqual(_retained_complete_laps(0, 0, 5), 0)
        self.assertEqual(_retained_complete_laps(1, 0, 5), 0)
        self.assertEqual(_retained_complete_laps(2, 0, 5), 1)
        self.assertEqual(_retained_complete_laps(6, 0, 5), 5)

    def test_standing_start_retains_first_closed_lap(self):
        self.assertEqual(
            _retained_complete_laps(0, 0, 5, standing_start=True), 0)
        self.assertEqual(
            _retained_complete_laps(1, 0, 5, standing_start=True), 1)
        self.assertEqual(
            _retained_complete_laps(5, 0, 5, standing_start=True), 5)


if __name__ == "__main__":
    unittest.main()
