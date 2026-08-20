"""V5 plan-first IL policy with a physically hard-capped visual residual."""

from __future__ import annotations

from .config import EGO_FEATURE_NAMES, PLANNING_HORIZONS_SECONDS, RACING_LINE_LOOKAHEADS_M
from .config_v5 import (
    BRAKE_RESIDUAL_CAP, STEER_RESIDUAL_CAP, TARGET_SPEED_WIDTH,
    THROTTLE_RESIDUAL_CAP,
)


def build_model_v5(pretrained: bool = False,
                   pred_len: int = len(PLANNING_HORIZONS_SECONDS)):
    import torch
    from torch import nn
    from torchvision.models import ResNet34_Weights, resnet34

    class TCPACV5(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
            line_width = len(RACING_LINE_LOOKAHEADS_M) * 2
            state_width = len(EGO_FEATURE_NAMES) + line_width + 6 + TARGET_SPEED_WIDTH
            self.measurements = nn.Sequential(
                nn.Linear(state_width, 192), nn.ReLU(inplace=True),
                nn.Linear(192, 192), nn.ReLU(inplace=True),
            )
            self.planner_join = nn.Sequential(
                nn.Linear(512 + 192, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
            )
            self.plan_gru = nn.GRUCell(3 + line_width, 256)
            self.waypoint_delta = nn.Linear(256, 2)
            self.future_speed_head = nn.Linear(256, 1)

            # This path receives only explicit predicted plan/state tensors.
            # There is intentionally no unrestricted visual feature input here.
            plan_width = pred_len * 3 + len(EGO_FEATURE_NAMES) + TARGET_SPEED_WIDTH + 6
            self.plan_control_net = nn.Sequential(
                nn.Linear(plan_width, 384), nn.ReLU(inplace=True),
                nn.Linear(384, 192), nn.ReLU(inplace=True),
                nn.Linear(192, 3),
            )
            step_width = 3 + len(EGO_FEATURE_NAMES) + TARGET_SPEED_WIDTH + 6
            self.future_control_net = nn.Sequential(
                nn.Linear(step_width, 192), nn.ReLU(inplace=True),
                nn.Linear(192, 96), nn.ReLU(inplace=True), nn.Linear(96, 3),
            )
            self.visual_residual_net = nn.Sequential(
                nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, 3),
            )
            self.pred_len = pred_len
            self.register_buffer(
                "residual_caps",
                torch.tensor([STEER_RESIDUAL_CAP, THROTTLE_RESIDUAL_CAP,
                              BRAKE_RESIDUAL_CAP], dtype=torch.float32))

        @staticmethod
        def _activate_control(raw):
            return torch.cat((raw[:, :1].tanh(), raw[:, 1:].sigmoid()), dim=1)

        def control_from_plan(self, waypoints, future_speeds, ego, target_speed, command):
            """Public for wiring tests: GT targets never enter this function."""
            explicit = torch.cat((
                waypoints.flatten(1), future_speeds.flatten(1), ego,
                target_speed, command), dim=1)
            current = self._activate_control(self.plan_control_net(explicit))
            future = []
            for step in range(self.pred_len):
                item = torch.cat((waypoints[:, step], future_speeds[:, step], ego,
                                  target_speed, command), dim=1)
                future.append(self._activate_control(self.future_control_net(item)))
            return current, torch.stack(future, dim=1)

        def forward(self, image, ego, racing_line, command, target_speed):
            if target_speed.ndim != 2 or target_speed.shape[1] != TARGET_SPEED_WIDTH:
                raise ValueError(f"target_speed width must be {TARGET_SPEED_WIDTH}")
            feature_map = self.encoder(image)
            visual = self.pool(feature_map).flatten(1)
            line_flat = racing_line.flatten(1)
            state = self.measurements(torch.cat(
                (ego, line_flat, command, target_speed), dim=1))
            hidden = self.planner_join(torch.cat((visual, state), dim=1))
            point = image.new_zeros((image.shape[0], 2))
            speed = image.new_zeros((image.shape[0], 1))
            waypoints, future_speeds = [], []
            for _ in range(self.pred_len):
                hidden = self.plan_gru(torch.cat((point, speed, line_flat), dim=1), hidden)
                point = point + self.waypoint_delta(hidden)
                speed = self.future_speed_head(hidden)
                waypoints.append(point)
                future_speeds.append(speed)
            waypoints = torch.stack(waypoints, dim=1)
            future_speeds = torch.stack(future_speeds, dim=1)
            plan_control, future_controls = self.control_from_plan(
                waypoints, future_speeds, ego, target_speed, command)

            residual = torch.tanh(self.visual_residual_net(visual)) * self.residual_caps
            final_control = torch.cat((
                (plan_control[:, :1] + residual[:, :1]).clamp(-1.0, 1.0),
                (plan_control[:, 1:] + residual[:, 1:]).clamp(0.0, 1.0),
            ), dim=1)
            # A coding error can never silently turn this into an unrestricted bypass.
            if self.training and torch.any(
                    residual.detach().abs() > self.residual_caps + 1e-6):
                raise RuntimeError("visual residual exceeded its hard cap")
            return {
                "waypoints": waypoints,
                "future_speeds": future_speeds,
                "plan_controls": plan_control,
                "visual_residual": residual,
                "controls": final_control,
                "future_controls": future_controls,
            }

    return TCPACV5()

