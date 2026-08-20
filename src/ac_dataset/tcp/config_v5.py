"""Frozen v5 schema.  Kept separate so v4 checkpoints remain reproducible."""

from __future__ import annotations

from .config import EGO_FEATURE_NAMES, PLANNING_HORIZONS_SECONDS, RACING_LINE_LOOKAHEADS_M

SCHEMA_VERSION = 5
PROFILE_BINS = 500
TARGET_SPEED_WIDTH = len(RACING_LINE_LOOKAHEADS_M) + 1
STEER_RESIDUAL_CAP = 0.10
THROTTLE_RESIDUAL_CAP = 0.05
BRAKE_RESIDUAL_CAP = 0.05


def model_schema_v5(profile_sha256: str, normalization_sha256: str = "") -> dict[str, object]:
    return {
        "version": SCHEMA_VERSION,
        "image_input": {"shape_chw": [3, 256, 512], "imagenet_normalized": True},
        "ego_features": list(EGO_FEATURE_NAMES),
        "racing_line": {
            "frame": "ego_local_right_forward",
            "lookaheads_m": list(RACING_LINE_LOOKAHEADS_M),
        },
        "target_speed_input": {
            "width": TARGET_SPEED_WIDTH,
            "lookaheads_m": list(RACING_LINE_LOOKAHEADS_M),
            "features": ["expert_speed_at_each_lookahead", "current_target_minus_speed"],
            "source": "train_normal_AI_actual_speed_median_by_spline",
            "profile_sha256": profile_sha256,
            "future_actual_speed_input": False,
        },
        "absolute_pose_input": False,
        "spline_position_model_input": False,
        "planning_outputs": ["waypoints", "future_speeds", "future_controls"],
        "planning_horizons_seconds": list(PLANNING_HORIZONS_SECONDS),
        "control_conditioning": "predicted_plan_only_plus_hard_capped_visual_residual",
        "teacher_forcing_control_plan": False,
        "controls": ["steer", "throttle", "brake"],
        "visual_residual_caps": {
            "steer": STEER_RESIDUAL_CAP,
            "throttle": THROTTLE_RESIDUAL_CAP,
            "brake": BRAKE_RESIDUAL_CAP,
        },
        "normalization_sha256": normalization_sha256,
    }

