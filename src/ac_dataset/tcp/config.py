"""Shared v2 input/output schema for training and live inference."""

POSITION_SCALE_M = 50.0
RACING_LINE_SCALE_M = 150.0
SPEED_SCALE_MPS = 90.0
LATERAL_SPEED_SCALE_MPS = 10.0
YAW_RATE_SCALE_RAD_S = 2.0
LONGITUDINAL_ACCEL_SCALE_MPS2 = 50.0
LATERAL_ACCEL_SCALE_MPS2 = 50.0
INPUT_SCHEMA_VERSION = 3
V4_INPUT_SCHEMA_VERSION = 4
RACING_LINE_LOOKAHEADS_M = (
    0.0, 5.0, 10.0, 20.0, 30.0, 40.0,
    50.0, 75.0, 100.0, 125.0, 150.0,
)
FUTURE_OFFSETS_FRAMES = (4, 10, 20, 40, 60, 80)
PLANNING_HORIZONS_SECONDS = tuple(offset / 40.0 for offset in FUTURE_OFFSETS_FRAMES)
EGO_FEATURE_NAMES = (
    "speed_mps",
    "longitudinal_speed_mps",
    "lateral_speed_mps",
    "yaw_rate_rad_s",
    "longitudinal_accel_mps2",
    "lateral_accel_mps2",
)
EGO_FEATURE_SCALES = (
    SPEED_SCALE_MPS,
    SPEED_SCALE_MPS,
    LATERAL_SPEED_SCALE_MPS,
    YAW_RATE_SCALE_RAD_S,
    LONGITUDINAL_ACCEL_SCALE_MPS2,
    LATERAL_ACCEL_SCALE_MPS2,
)
EGO_FEATURE_NORMALIZED_CLIP = 4.0
CONTROL_NAMES = ("steer", "throttle", "brake")

# The fixed-route command remains at index 3.  The two previously unused
# command slots explicitly disambiguate the standing-start lap from flying
# laps without changing the model input shape.
STANDING_LAP_COMMAND_INDEX = 4
FLYING_LAP_COMMAND_INDEX = 5


def normalization_metadata(target_speed_input: bool = False) -> dict[str, object]:
    payload = {
        "schema_version": V4_INPUT_SCHEMA_VERSION if target_speed_input else INPUT_SCHEMA_VERSION,
        "position_scale_m": POSITION_SCALE_M,
        "racing_line_scale_m": RACING_LINE_SCALE_M,
        "speed_scale_mps": SPEED_SCALE_MPS,
        "ego_features": {
            name: scale for name, scale in zip(EGO_FEATURE_NAMES, EGO_FEATURE_SCALES)
        },
        "ego_normalized_clip": EGO_FEATURE_NORMALIZED_CLIP,
        "racing_line_lookaheads_m": list(RACING_LINE_LOOKAHEADS_M),
        "image": {
            "color": "RGB",
            "range": "uint8_to_0_1",
            "resize_hw": [256, 512],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    if target_speed_input:
        payload["target_speed"] = {
            "lookaheads_m": list(RACING_LINE_LOOKAHEADS_M),
            "scale_mps": SPEED_SCALE_MPS,
            "features": ["profile_speed_at_each_lookahead", "current_speed_error"],
            "source": "AI_RACING_LINE.csv",
        }
    return payload


def model_schema_metadata(target_speed_input: bool = False) -> dict[str, object]:
    payload = {
        "version": V4_INPUT_SCHEMA_VERSION if target_speed_input else INPUT_SCHEMA_VERSION,
        "ego_features": list(EGO_FEATURE_NAMES),
        "racing_line": {
            "frame": "ego_local_right_forward",
            "lookaheads_m": list(RACING_LINE_LOOKAHEADS_M),
            "scale_m": RACING_LINE_SCALE_M,
        },
        "absolute_pose_input": False,
        "controls": list(CONTROL_NAMES),
        "planning_outputs": ["waypoints", "future_controls"],
        "control_conditioning": "predicted_waypoints_and_trajectory_states",
        "planning_horizons_seconds": list(PLANNING_HORIZONS_SECONDS),
    }
    if target_speed_input:
        payload["target_speed_input"] = {
            "width": len(RACING_LINE_LOOKAHEADS_M) + 1,
            "profile_points": len(RACING_LINE_LOOKAHEADS_M),
            "includes_current_speed_error": True,
            "future_actual_speed_input": False,
        }
        payload["partial_future_targets"] = True
        payload["recovery_episode_boundaries"] = True
    return payload


def target_speed_input_width(schema: dict[str, object] | None) -> int:
    section = {} if schema is None else schema.get("target_speed_input", {})
    return int(section.get("width", 0)) if isinstance(section, dict) else 0


def lap_phase_metadata() -> dict[str, object]:
    return {
        "encoding": "one_hot_command",
        "standing_lap_command_index": STANDING_LAP_COMMAND_INDEX,
        "flying_lap_command_index": FLYING_LAP_COMMAND_INDEX,
        "standing_dataset_lap": 1,
        "flying_dataset_lap_min": 2,
    }
