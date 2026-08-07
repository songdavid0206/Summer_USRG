"""Shared physical scales for TCP-AC training and closed-loop inference."""

POSITION_SCALE_M = 50.0
SPEED_SCALE_MPS = 90.0

# The fixed-route command remains at index 3.  The two previously unused
# command slots explicitly disambiguate the standing-start lap from flying
# laps without changing the model input shape.
STANDING_LAP_COMMAND_INDEX = 4
FLYING_LAP_COMMAND_INDEX = 5


def normalization_metadata() -> dict[str, float]:
    return {
        "position_scale_m": POSITION_SCALE_M,
        "speed_scale_mps": SPEED_SCALE_MPS,
    }


def lap_phase_metadata() -> dict[str, object]:
    return {
        "encoding": "one_hot_command",
        "standing_lap_command_index": STANDING_LAP_COMMAND_INDEX,
        "flying_lap_command_index": FLYING_LAP_COMMAND_INDEX,
        "standing_dataset_lap": 1,
        "flying_dataset_lap_min": 2,
    }
