import numpy as np
import unittest

from ac_dataset.tcp.config_v5 import (
    BRAKE_RESIDUAL_CAP, STEER_RESIDUAL_CAP, TARGET_SPEED_WIDTH,
    THROTTLE_RESIDUAL_CAP,
)


class TCPV5Tests(unittest.TestCase):
    def test_residual_caps_and_predicted_plan_wiring(self):
        import torch
        from ac_dataset.tcp.model_v5 import build_model_v5

        model = build_model_v5(False).eval()
        batch = 1
        with torch.no_grad():
            result = model(
                torch.zeros(batch, 3, 64, 128),
                torch.zeros(batch, 6),
                torch.zeros(batch, 11, 2),
                torch.zeros(batch, 6),
                torch.zeros(batch, TARGET_SPEED_WIDTH),
            )
        caps = np.asarray(
            [STEER_RESIDUAL_CAP, THROTTLE_RESIDUAL_CAP, BRAKE_RESIDUAL_CAP])
        self.assertTrue(np.all(
            result["visual_residual"].abs().numpy() <= caps + 1e-6))
        self.assertEqual(result["waypoints"].shape, (batch, 6, 2))
        self.assertEqual(result["future_speeds"].shape, (batch, 6, 1))
        self.assertEqual(result["plan_controls"].shape, (batch, 3))
        self.assertEqual(result["controls"].shape, (batch, 3))

    def test_waypoint_speed_consistency_uses_segment_average(self):
        import torch
        from ac_dataset.tcp.train_v5 import compute_losses

        horizons = torch.tensor([.1, .25, .5, 1., 1.5, 2.])
        speeds = torch.tensor([[12., 15., 20., 25., 30., 35.]])
        previous = torch.cat((torch.tensor([[10.]]), speeds[:, :-1]), dim=1)
        dt = torch.cat((horizons[:1], horizons[1:] - horizons[:-1]))
        distance = (.5 * (previous + speeds) * dt).cumsum(dim=1)
        waypoints = torch.stack((torch.zeros_like(distance), distance), dim=2)
        pred = {
            "waypoints": waypoints, "future_speeds": speeds.unsqueeze(-1),
            "visual_residual": torch.zeros(1, 3),
        }
        batch = {
            "waypoints_m": waypoints.clone(),
            "future_speeds_mps": speeds.unsqueeze(-1).clone(),
            "future_mask": torch.ones(1, 6), "ego_raw": torch.tensor([[10., 0, 0, 0, 0, 0]]),
        }
        stats = {
            "waypoints_local_m": (torch.zeros(6, 2), torch.ones(6, 2)),
            "future_speed_mps": (torch.zeros(6), torch.ones(6)),
        }
        _, metrics = compute_losses(pred, batch, stats, planner_only=True)
        self.assertLess(float(metrics["consistency"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
