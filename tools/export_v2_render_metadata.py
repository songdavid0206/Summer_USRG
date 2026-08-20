"""Attach raw render-camera metadata to selected dataset_v2 sessions.

The structured export resamples/renumbers frames, while the raw camera and
render-car parquet files use capture frame numbers.  This tool maps both raw
tables through ``FRAME_INDEX.csv`` and writes v2-numbered tables at the session
root.  Original identifiers are retained as ``source_*`` columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CAMERA_COLUMNS = [
    "csp_bridge_version",
    "cam_pose_seq",
    "cam_render_frame",
    "cam_sim_time_ms",
    "cam_game_time_ms",
    "cam_lap_count",
    "cam_pose_game_delta_ms",
    "cam_packet_render_offset",
    "cam_packet_render_error",
    "cam_pos_x",
    "cam_pos_y",
    "cam_pos_z",
    "cam_forward_x",
    "cam_forward_y",
    "cam_forward_z",
    "cam_up_x",
    "cam_up_y",
    "cam_up_z",
    "cam_fov_deg",
    "cam_mode",
    "cam_drivable_mode",
    "cam_car_index",
    "cam_focused_car",
]


def find_session(v2_root: Path, run_id: str) -> Path:
    matches = [
        path
        for mode in ("solo", "multi")
        for path in (v2_root / mode).glob(f"{run_id}_*")
        if path.is_dir()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one dataset_v2 directory for {run_id}, found {matches}"
        )
    return matches[0]


def export_session(raw_root: Path, session: Path, run_id: str) -> dict[str, int]:
    raw = raw_root / run_id
    raw_image_path = raw / "image_index.parquet"
    raw_render_path = raw / "render_car_state.parquet"
    for required in (session / "FRAME_INDEX.csv", raw_image_path, raw_render_path):
        if not required.exists():
            raise FileNotFoundError(required)

    mapping = pd.read_csv(session / "FRAME_INDEX.csv")
    if mapping["time_ms"].duplicated().any():
        raise ValueError(f"{session.name}: duplicate session-level time_ms")
    mapping = mapping.copy()
    mapping.insert(0, "frame", np.arange(len(mapping), dtype=np.int64))
    mapping_columns = [
        column
        for column in (
            "frame",
            "time_ms",
            "dataset_lap_number",
            "source_time_ms",
            "source_index",
            "source_frame",
            "t_img",
            "lap_count",
            "spline_position",
        )
        if column in mapping.columns
    ]
    mapping = mapping[mapping_columns]

    raw_images = pd.read_parquet(raw_image_path)
    raw_images = raw_images.rename(
        columns={"frame": "source_frame_raw", "image_path": "source_image_path"}
    )
    camera_columns = [column for column in CAMERA_COLUMNS if column in raw_images.columns]
    camera_source = raw_images[
        ["source_frame_raw", "source_image_path", *camera_columns]
    ]
    if camera_source["source_frame_raw"].duplicated().any():
        raise ValueError(f"{run_id}: duplicate raw image frame")

    image_index = mapping.merge(
        camera_source,
        left_on="source_frame",
        right_on="source_frame_raw",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing_camera = int((image_index["_merge"] != "both").sum())
    image_index = image_index.drop(columns=["_merge", "source_frame_raw"])
    if missing_camera:
        raise ValueError(f"{session.name}: {missing_camera} v2 frames lack raw camera pose")

    raw_render = pd.read_parquet(raw_render_path).rename(
        columns={"image_frame": "source_image_frame"}
    )
    render_mapping = mapping.rename(columns={"source_frame": "source_image_frame"})
    render_state = render_mapping.merge(
        raw_render,
        on="source_image_frame",
        how="left",
        validate="many_to_many",
        indicator=True,
    )
    missing_render = int((render_state["_merge"] != "both").sum())
    if missing_render:
        raise ValueError(f"{session.name}: {missing_render} v2 frames lack render-car state")
    render_state = render_state.drop(columns="_merge")
    render_state.insert(
        render_state.columns.get_loc("source_image_frame"),
        "image_frame",
        render_state["frame"].astype(np.int64),
    )

    image_index.to_parquet(session / "image_index.parquet", index=False)
    render_state.to_parquet(session / "render_car_state.parquet", index=False)

    info = (
        "schema=v2_remapped_render_metadata\n"
        f"run_id={run_id}\n"
        f"v2_frames={len(image_index)}\n"
        f"render_car_rows={len(render_state)}\n"
        "join=image_index.frame = render_car_state.image_frame\n"
        "image_lookup_solo=dataset_lap_number + time_ms\n"
        "image_lookup_nested_multi=clip FRAME_INDEX.source_frame = image_index.source_frame\n"
        "source_trace=source_frame/source_image_frame refer to raw capture frame\n"
        "coordinate_frame=(x,y,z)=(AC z,AC x,AC y)\n"
    )
    (session / "RENDER_METADATA_INFO.txt").write_text(info, encoding="utf-8")
    return {
        "frames": len(image_index),
        "render_rows": len(render_state),
        "cars": int(render_state["car_id"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("dataset"))
    parser.add_argument("--v2-root", type=Path, default=Path("dataset_v2"))
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()

    for run_id in args.run_ids:
        session = find_session(args.v2_root, run_id)
        stats = export_session(args.raw_root, session, run_id)
        print(
            f"[ok] {session.name}: frames={stats['frames']} "
            f"render_rows={stats['render_rows']} cars={stats['cars']}"
        )


if __name__ == "__main__":
    main()
