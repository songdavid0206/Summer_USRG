"""TCP-style trajectory-guided control network adapted to AC labels."""

from __future__ import annotations

from .config import SPEED_SCALE_MPS


def build_model(pretrained: bool = False, pred_len: int = 4):
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
            self.measurements = nn.Sequential(
                nn.Linear(1 + 2 + 6, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
            )
            self.traj_join = nn.Sequential(
                nn.Linear(512 + 128, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
            )
            self.traj_gru = nn.GRUCell(4, 256)
            self.wp_head = nn.Linear(256, 2)
            self.ctrl_join = nn.Sequential(
                nn.Linear(512 + 128, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
            )
            self.ctrl_gru = nn.GRUCell(256 + 2, 256)
            self.action_head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, 2), nn.Tanh())
            self.speed_head = nn.Linear(512, 1)
            self.pred_len = pred_len

        def forward(self, image, speed, target_point, command):
            feature_map = self.encoder(image)
            visual = self.pool(feature_map).flatten(1)
            state = self.measurements(torch.cat(
                [speed / SPEED_SCALE_MPS, target_point, command], dim=1,
            ))

            hidden_traj = self.traj_join(torch.cat([visual, state], dim=1))
            point = torch.zeros((image.shape[0], 2), dtype=image.dtype, device=image.device)
            waypoints = []
            trajectory_states = []
            for _ in range(self.pred_len):
                hidden_traj = self.traj_gru(torch.cat([point, target_point], dim=1), hidden_traj)
                point = point + self.wp_head(hidden_traj)
                waypoints.append(point)
                trajectory_states.append(hidden_traj)

            ctrl_state = self.ctrl_join(torch.cat([visual, state], dim=1))
            current_action = self.action_head(ctrl_state)
            hidden_ctrl = torch.zeros_like(ctrl_state)
            future_actions = []
            for step in range(self.pred_len):
                hidden_ctrl = self.ctrl_gru(torch.cat([trajectory_states[step], current_action], dim=1), hidden_ctrl)
                current_action = self.action_head(hidden_ctrl)
                future_actions.append(current_action)
            return {
                "waypoints": torch.stack(waypoints, dim=1),
                "action": self.action_head(ctrl_state),
                "future_actions": torch.stack(future_actions, dim=1),
                "speed": self.speed_head(visual),
            }

    return TCPAC()
