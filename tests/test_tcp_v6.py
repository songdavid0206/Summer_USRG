import unittest


class TCPV6Tests(unittest.TestCase):
    def test_v7_control_alignment_uses_future_same_trajectory_only(self):
        import pandas as pd
        from ac_dataset.tcp.prepare_v7 import align_control_labels

        table = pd.DataFrame({
            "trajectory_id": ["a", "a", "a", "a", "b", "b"],
            "time_ms": [0, 25, 50, 75, 0, 25],
            "steer": [0., .1, .2, .3, .8, .9],
            "throttle": [0., .1, .2, .3, .8, .9],
            "brake": [0., .1, .2, .3, .8, .9],
            "pedal_loss_weight": [1.] * 6,
            "eligible": [True] * 6,
            "sampling_bucket": ["dagger_mild"] * 6,
            "source_type": ["dagger"] * 6,
        })
        aligned, summary = align_control_labels(table, delay_frames=2)
        self.assertAlmostEqual(aligned.loc[0, "steer"], .2)
        self.assertAlmostEqual(aligned.loc[1, "steer"], .3)
        self.assertFalse(bool(aligned.loc[2, "eligible"]))
        self.assertFalse(bool(aligned.loc[4, "eligible"]))
        self.assertEqual(summary["delay_ms"], 50)
        self.assertFalse(summary["trajectory_boundary_crossing_allowed"])

    def test_future_control_mask_does_not_remove_planner_horizon(self):
        import torch
        from ac_dataset.tcp.train_v5 import compute_losses

        pred = {
            "waypoints": torch.zeros(1, 6, 2),
            "future_speeds": torch.zeros(1, 6, 1),
            "controls": torch.zeros(1, 3),
            "future_controls": torch.ones(1, 6, 3),
            "visual_residual": torch.zeros(1, 3),
        }
        batch = {
            "waypoints_m": torch.zeros(1, 6, 2),
            "future_speeds_mps": torch.zeros(1, 6, 1),
            "future_mask": torch.ones(1, 6),
            "future_control_mask": torch.tensor([[1., 1., 1., 1., 1., 0.]]),
            "controls": torch.zeros(1, 3),
            "future_controls": torch.zeros(1, 6, 3),
            "pedal_loss_weight": torch.ones(1),
            "future_pedal_loss_weight": torch.ones(1, 6),
            "ego_raw": torch.zeros(1, 6),
        }
        stats = {
            "waypoints_local_m": (torch.zeros(6, 2), torch.ones(6, 2)),
            "future_speed_mps": (torch.zeros(6), torch.ones(6)),
        }
        _, metrics = compute_losses(pred, batch, stats, planner_only=False)
        self.assertTrue(torch.isfinite(metrics["future_control"]))
        self.assertAlmostEqual(float(metrics["future_control"]), 1.0)

    def test_future_only_perturbation_leaves_near_targets_and_error_unchanged(self):
        import torch
        from ac_dataset.tcp.config_v5 import TARGET_SPEED_WIDTH
        from ac_dataset.tcp.train_v6 import lowered_target

        target = torch.zeros(2, TARGET_SPEED_WIDTH)
        stats = {"target_speed_features_mps": (
            torch.zeros(TARGET_SPEED_WIDTH), torch.ones(TARGET_SPEED_WIDTH))}
        changed = lowered_target(target, stats, 10., future_only=True)
        self.assertTrue(torch.equal(changed[:, :3], target[:, :3]))
        self.assertTrue(torch.equal(changed[:, -1:], target[:, -1:]))
        self.assertTrue(torch.all(changed[:, 3:-1] == -10.))

    def test_v5_planner_weights_are_exactly_compatible(self):
        from ac_dataset.tcp.model_v5 import build_model_v5
        from ac_dataset.tcp.model_v6 import build_model_v6

        v5 = build_model_v5(False)
        v6 = build_model_v6(False)
        result = v6.load_state_dict(v5.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_counterfactual_loss_rejects_inverse_pedals(self):
        import torch
        from ac_dataset.tcp.train_v6 import causal_pair_metrics

        baseline = {
            "future_speeds": torch.zeros(2, 6, 1),
            "plan_controls": torch.tensor([[0., .4, .2], [0., .4, .2]]),
            "controls": torch.tensor([[0., .4, .2], [0., .4, .2]]),
        }
        good = {
            "future_speeds": torch.full((2, 6, 1), -1.),
            "plan_controls": torch.tensor([[0., .3, .3], [0., .3, .3]]),
            "controls": torch.tensor([[0., .3, .3], [0., .3, .3]]),
        }
        bad = {
            "future_speeds": torch.full((2, 6, 1), -1.),
            "plan_controls": torch.tensor([[0., .5, .1], [0., .5, .1]]),
            "controls": torch.tensor([[0., .5, .1], [0., .5, .1]]),
        }
        batch = {"future_mask": torch.ones(2, 6)}
        stats = {"future_speed_mps": (torch.zeros(6), torch.ones(6))}
        good_loss, _ = causal_pair_metrics(baseline, good, batch, stats)
        bad_loss, _ = causal_pair_metrics(baseline, bad, batch, stats)
        self.assertLess(float(good_loss), 1e-7)
        self.assertGreater(float(bad_loss), .05)

    def test_full_gate_requires_direction_magnitude_and_coverage(self):
        from ac_dataset.tcp.train_v6 import full_gate_passed

        probe = {
            "future_speed_delta_mps": -.8,
            "future_speed_monotonic_fraction": .9,
            "plan_throttle_delta": -.03,
            "plan_throttle_monotonic_fraction": .8,
            "plan_brake_delta": .03,
            "plan_brake_monotonic_fraction": .8,
            "plan_longitudinal_delta": -.12,
            "plan_longitudinal_monotonic_fraction": .8,
            "final_throttle_delta": -.03,
            "final_throttle_monotonic_fraction": .8,
            "final_brake_delta": .03,
            "final_brake_monotonic_fraction": .8,
            "final_longitudinal_delta": -.12,
            "final_longitudinal_monotonic_fraction": .8,
            "residual_change_abs_max": 0.,
        }
        self.assertTrue(full_gate_passed(probe))
        probe["final_throttle_delta"] = .01
        self.assertFalse(full_gate_passed(probe))

    def test_full_gate_rejects_one_failed_source(self):
        from copy import deepcopy
        from ac_dataset.tcp.train_v6 import full_gate_passed

        passing = {
            "future_speed_delta_mps": -.8,
            "future_speed_monotonic_fraction": .9,
            "plan_throttle_delta": -.03,
            "plan_throttle_monotonic_fraction": .8,
            "plan_brake_delta": .03,
            "plan_brake_monotonic_fraction": .8,
            "plan_longitudinal_delta": -.12,
            "plan_longitudinal_monotonic_fraction": .8,
            "final_throttle_delta": -.03,
            "final_throttle_monotonic_fraction": .8,
            "final_brake_delta": .03,
            "final_brake_monotonic_fraction": .8,
            "final_longitudinal_delta": -.12,
            "final_longitudinal_monotonic_fraction": .8,
            "residual_change_abs_max": 0.,
        }
        probe = deepcopy(passing)
        probe["by_source"] = {"normal": deepcopy(passing), "dagger": deepcopy(passing)}
        probe["by_source"]["dagger"]["plan_brake_delta"] = -.01
        self.assertFalse(full_gate_passed(probe))

    def test_full_gate_rejects_failed_future_anticipation(self):
        from copy import deepcopy
        from ac_dataset.tcp.train_v6 import full_gate_passed

        passing = {
            "future_speed_delta_mps": -.8,
            "future_speed_monotonic_fraction": .9,
            "plan_throttle_delta": -.06,
            "plan_throttle_monotonic_fraction": .8,
            "plan_brake_delta": .06,
            "plan_brake_monotonic_fraction": .8,
            "plan_longitudinal_delta": -.12,
            "plan_longitudinal_monotonic_fraction": .8,
            "final_throttle_delta": -.06,
            "final_throttle_monotonic_fraction": .8,
            "final_brake_delta": .06,
            "final_brake_monotonic_fraction": .8,
            "final_longitudinal_delta": -.12,
            "final_longitudinal_monotonic_fraction": .8,
            "residual_change_abs_max": 0.,
        }
        probe = deepcopy(passing)
        probe["anticipation"] = deepcopy(passing)
        probe["anticipation"]["final_longitudinal_delta"] = -.01
        self.assertFalse(full_gate_passed(probe))


if __name__ == "__main__":
    unittest.main()
