import unittest

import numpy as np
import pandas as pd

from ac_dataset.tcp.config import (
    model_schema_metadata, normalization_metadata, target_speed_input_width,
)
from ac_dataset.tcp.shadow import RacingLineTarget
from ac_dataset.tcp.train import losses


class TCPV4Tests(unittest.TestCase):
    def test_v4_schema_adds_only_planned_speed_measurements(self):
        schema = model_schema_metadata(True)
        self.assertEqual(schema["version"], 4)
        self.assertEqual(target_speed_input_width(schema), 12)
        self.assertFalse(schema["target_speed_input"]["future_actual_speed_input"])
        self.assertEqual(normalization_metadata(True)["schema_version"], 4)

    def test_live_target_speed_uses_same_lookahead_search(self):
        line = pd.DataFrame({
            "normalized_position": [0, .25, .5, .75],
            "x": [0, 25, 50, 75], "y": [0, 0, 0, 0],
            "dist": [0, 25, 50, 75], "ai_speed_ms": [10, 20, 30, 40],
        })
        provider = RacingLineTarget(line)
        profile, features = provider.target_speed_features(0, 5)
        self.assertEqual(len(profile), 11)
        self.assertAlmostEqual(profile[0], 10)
        self.assertAlmostEqual(features[-1], 5 / 90)

    def test_masked_future_garbage_does_not_change_loss(self):
        import torch
        pred = {
            "controls": torch.zeros(1, 3), "waypoints": torch.zeros(1, 2, 2),
            "future_controls": torch.zeros(1, 2, 3), "speed": torch.zeros(1, 1),
        }
        batch = {
            "controls": torch.zeros(1, 3), "waypoints": torch.tensor([[[0., 0.], [999., 999.]]]),
            "future_controls": torch.tensor([[[0., 0., 0.], [999., 999., 999.]]]),
            "future_mask": torch.tensor([[1., 0.]]),
            "pedal_loss_weight": torch.ones(1),
            "future_pedal_loss_weight": torch.ones(1, 2), "speed": torch.zeros(1, 1),
        }
        total, parts = losses(pred, batch, speed_weight=0)
        self.assertEqual(float(total), 0.0)
        self.assertEqual(float(parts["waypoint"]), 0.0)


if __name__ == "__main__":
    unittest.main()
