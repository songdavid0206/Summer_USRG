import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ac_dataset.validate import _check_collisions, _game_alignment


class GameAlignmentTest(unittest.TestCase):
    def test_wall_clock_delay_does_not_fail_game_time_alignment(self):
        cam = pd.DataFrame({
            "game_time_ms": [1000, 2000],
            "completed_laps": [1, 1],
        })
        ego = pd.DataFrame({
            "currentTime": [1002, 1998],
            "completedLaps": [1, 1],
            "NormalizedSplinePosition": [.2, .4],
            "dt": [.2, .3],
        })
        _, boundary, bad = _game_alignment(cam, ego, 50)
        self.assertFalse(boundary.any())
        self.assertFalse(bad.any())

    def test_async_finish_line_reset_is_separated_from_regular_errors(self):
        cam = pd.DataFrame({
            "game_time_ms": [3, 5000],
            "completed_laps": [3, 3],
        })
        ego = pd.DataFrame({
            "currentTime": [69000, 5200],
            "completedLaps": [2, 3],
            "NormalizedSplinePosition": [.001, .5],
        })
        _, boundary, bad = _game_alignment(cam, ego, 50)
        self.assertEqual(boundary.tolist(), [True, False])
        self.assertEqual(bad.tolist(), [False, True])


class CollisionIdAuditTest(unittest.TestCase):
    def test_collision_with_is_decoded_as_car_id_plus_one(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "camera").mkdir()
            opponent_dir = run / "processed" / "opponent"
            opponent_dir.mkdir(parents=True)
            pd.DataFrame([{
                "t_img": 1.0,
                "collision_with": 3,
                "collision_position_up": .25,
            }]).to_parquet(run / "camera" / "collision_events.parquet")
            pd.DataFrame({
                "t_img": [1.0], "distance": [30.0],
            }).to_parquet(opponent_dir / "opp_1_features.parquet")
            pd.DataFrame({
                "t_img": [1.0], "distance": [4.8],
            }).to_parquet(opponent_dir / "opp_2_features.parquet")

            checks = {check.name: check for check in _check_collisions(run)}

            self.assertTrue(checks["collision_id_mapping_ok"].passed)
            self.assertIn("1/1 valid", checks["collision_id_mapping_ok"].detail)


if __name__ == "__main__":
    unittest.main()
