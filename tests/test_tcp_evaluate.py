import unittest

import pandas as pd

from ac_dataset.tcp.evaluate import (
    HORIZONS, prediction_metrics, qualitative_indices, training_summary,
)


class TCPEvaluateTests(unittest.TestCase):
    def test_training_summary_selects_lowest_validation_total(self):
        history = [
            {"epoch": 1, "val": {"total": 2.0, "action": 1.0, "waypoint": 1.0,
                                   "future_action": 1.0, "speed": 1.0}},
            {"epoch": 2, "val": {"total": 1.0, "action": .5, "waypoint": .5,
                                   "future_action": .5, "speed": .5}},
        ]
        summary = training_summary(history)
        self.assertEqual(summary["best_epoch"], 2)
        self.assertEqual(summary["last_epoch"], 2)

    def test_prediction_metrics_reconstructs_weighted_loss(self):
        rows = {"gt_steer": [0.0], "pred_steer": [.4],
                "gt_throttle": [0.0], "pred_throttle": [.2],
                "gt_brake": [0.0], "pred_brake": [.1],
                "speed_mps": [0.0], "pred_speed_mps": [9.0]}
        for step in range(1, len(HORIZONS) + 1):
            rows[f"gt_wp{step}_lateral_m"] = [0.0]
            rows[f"pred_wp{step}_lateral_m"] = [5.0]
            rows[f"gt_wp{step}_forward_m"] = [0.0]
            rows[f"pred_wp{step}_forward_m"] = [5.0]
            rows[f"gt_future{step}_steer"] = [0.0]
            rows[f"pred_future{step}_steer"] = [.4]
            rows[f"gt_future{step}_throttle"] = [0.0]
            rows[f"pred_future{step}_throttle"] = [.2]
            rows[f"gt_future{step}_brake"] = [0.0]
            rows[f"pred_future{step}_brake"] = [.1]
        checkpoint = {"epoch": 1, "val_loss": 1.155,
                      "normalization": {}}
        metrics = prediction_metrics(pd.DataFrame(rows), checkpoint)
        # .7 controls + .1 waypoint + .5*.7 future + .05*.1 speed
        self.assertAlmostEqual(metrics["recomputed_validation_loss"]["total"], 1.155)

    def test_qualitative_indices_are_valid(self):
        table = pd.DataFrame({"combined_error_score": [3.0, 1.0, 2.0],
                              "spline_position": [.3, .1, .2], "speed_mps": [30, 10, 20],
                              "session": ["a", "a", "a"]})
        selected = qualitative_indices(table, 2)
        self.assertEqual(selected["best"], [1, 2])
        self.assertEqual(selected["worst"], [0, 2])
        self.assertTrue(all(0 <= i < len(table) for values in selected.values() for i in values))


if __name__ == "__main__":
    unittest.main()
