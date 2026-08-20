"""TCP-style trajectory-guided control network adapted to AC labels."""

from __future__ import annotations

from .config import (
    EGO_FEATURE_NAMES, PLANNING_HORIZONS_SECONDS, RACING_LINE_LOOKAHEADS_M,
)


def build_model(pretrained: bool = False,
                pred_len: int = len(PLANNING_HORIZONS_SECONDS),
                target_speed_width: int = 0):
    import torch
    from torch import nn
    from torchvision.models import ResNet34_Weights, resnet34

    class TCPAC(nn.Module):
        def __init__(self):
            super().__init__()
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
            line_width = len(RACING_LINE_LOOKAHEADS_M) * 2
            self.target_speed_width = int(target_speed_width)
            measurement_width = (
                len(EGO_FEATURE_NAMES) + line_width + 6 + self.target_speed_width)
            self.measurements = nn.Sequential(
                nn.Linear(measurement_width, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
            )
            self.traj_join = nn.Sequential(
                nn.Linear(512 + 128, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
            )
            self.traj_gru = nn.GRUCell(2 + line_width, 256)
            self.wp_head = nn.Linear(256, 2)
            self.ctrl_join = nn.Sequential(
                nn.Linear(512 + 128 + pred_len * 2, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
            )
            self.ctrl_gru = nn.GRUCell(256 + 3, 256)
            self.control_features = nn.Sequential(
                nn.Linear(256, 256), nn.ReLU(inplace=True))
            self.control_output = nn.Linear(256, 3)
            self.speed_head = nn.Linear(512, 1)
            self.pred_len = pred_len

        def control_head(self, hidden):
            raw = self.control_output(self.control_features(hidden))
            # Model schema order: steer [-1,1], throttle [0,1], brake [0,1].
            return torch.cat((raw[:, :1].tanh(), raw[:, 1:].sigmoid()), dim=1)

        def forward(self, image, ego, racing_line, command, target_speed=None):
            feature_map = self.encoder(image)
            visual = self.pool(feature_map).flatten(1)
            line_flat = racing_line.flatten(1)
            measurements = [ego, line_flat, command]
            if self.target_speed_width:
                if target_speed is None or target_speed.shape[1] != self.target_speed_width:
                    raise ValueError(
                        f"target_speed width must be {self.target_speed_width}")
                measurements.append(target_speed)
            state = self.measurements(torch.cat(measurements, dim=1))

            hidden_traj = self.traj_join(torch.cat([visual, state], dim=1))
            point = torch.zeros((image.shape[0], 2), dtype=image.dtype, device=image.device)
            waypoints = []
            trajectory_states = []
            for _ in range(self.pred_len):
                hidden_traj = self.traj_gru(
                    torch.cat([point, line_flat], dim=1), hidden_traj)
                point = point + self.wp_head(hidden_traj)
                waypoints.append(point)
                trajectory_states.append(hidden_traj)

            waypoint_tensor = torch.stack(waypoints, dim=1)
            # Current controls explicitly consume the learned plan. This keeps
            # the planning head observable while letting its geometry guide actuation.
            ctrl_state = self.ctrl_join(torch.cat(
                [visual, state, waypoint_tensor.flatten(1)], dim=1))
            current_control = self.control_head(ctrl_state)
            hidden_ctrl = torch.zeros_like(ctrl_state)
            future_controls = []
            for step in range(self.pred_len):
                hidden_ctrl = self.ctrl_gru(
                    torch.cat([trajectory_states[step], current_control], dim=1),
                    hidden_ctrl)
                current_control = self.control_head(hidden_ctrl)
                future_controls.append(current_control)
            return {
                "waypoints": waypoint_tensor,
                "controls": self.control_head(ctrl_state),
                "future_controls": torch.stack(future_controls, dim=1),
                "speed": self.speed_head(visual),
            }

    return TCPAC()
