"""실제 렌더 카메라 공유메모리/라벨 저장 오프라인 테스트."""

import ctypes
import mmap
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ac_dataset.ac_shmem import (ACCameraPoseShm, CAMERA_POSE_CAPACITY,
                                 CAMERA_POSE_MAGIC, CAMERA_POSE_VERSION,
                                 CameraPoseRing)
from ac_dataset.recorder.image_logger import ImageLogger
from ac_dataset.recorder.dataset_builder import build_dataset


class CameraPoseTests(unittest.TestCase):
    def test_ring_selects_same_lap_and_nearest_game_time(self):
        size = ctypes.sizeof(CameraPoseRing)
        mm = mmap.mmap(0, size, "AcTools.ACDataset.RenderState.v3")
        ring = CameraPoseRing.from_buffer(mm)
        ring.magic, ring.version = CAMERA_POSE_MAGIC, CAMERA_POSE_VERSION
        ring.capacity, ring.write_index = CAMERA_POSE_CAPACITY, 2
        for i, lap_ms in enumerate((1000, 1020)):
            ring.slot_seq[i] = 2 * (i + 1)
            ring.render_frame[i] = 100 + i
            ring.lap_time_ms[i] = lap_ms
            ring.lap_count[i] = 3
            ring.pos_x[i], ring.pos_y[i], ring.pos_z[i] = 1 + i, 2, 3
            ring.forward_x[i], ring.forward_y[i], ring.forward_z[i] = 1, 0, 0
            ring.up_x[i], ring.up_y[i], ring.up_z[i] = 0, 0, 1
            ring.fov_deg[i] = 56
        ring.car_count[1] = 1
        k = 32  # slot 1 * RENDER_CAR_MAX + car 0
        ring.car_timestamp_ms[k] = 1018
        ring.car_pos_x[k], ring.car_pos_y[k], ring.car_pos_z[k] = 4, 5, 6
        ring.car_forward_x[k], ring.car_forward_y[k], ring.car_forward_z[k] = 1, 0, 0
        ring.car_up_x[k], ring.car_up_y[k], ring.car_up_z[k] = 0, 0, 1
        ring.car_aabb_length[k], ring.car_aabb_width[k], ring.car_aabb_height[k] = 4, 2, 1
        reader = ACCameraPoseShm()
        # Learn the synthetic session relation: packet = 2 * render - 61.
        learned = reader.read(1000, 3, graphics_packet=139)
        self.assertEqual(learned["cam_packet_render_offset"], -61)
        got = reader.read(1017, 3)
        self.assertEqual(got["cam_render_frame"], 101)
        self.assertEqual(got["cam_pose_game_delta_ms"], 3)
        self.assertEqual(got["_render_cars"][0]["car_pos_x"], 4.0)
        reader.close()
        del ring
        mm.close()

    def test_reader_returns_each_collision_event_once(self):
        size = ctypes.sizeof(CameraPoseRing)
        mm = mmap.mmap(-1, size)
        ring = CameraPoseRing.from_buffer(mm)
        ring.magic, ring.version = CAMERA_POSE_MAGIC, CAMERA_POSE_VERSION
        ring.capacity, ring.write_index = CAMERA_POSE_CAPACITY, 1
        ring.slot_seq[0] = 2
        ring.render_frame[0], ring.lap_time_ms[0], ring.lap_count[0] = 100, 1000, 0
        ring.pos_x[0], ring.pos_y[0], ring.pos_z[0] = 0, 0, 1
        ring.forward_x[0], ring.forward_y[0], ring.forward_z[0] = 1, 0, 0
        ring.up_x[0], ring.up_y[0], ring.up_z[0] = 0, 0, 1
        ring.fov_deg[0] = 60

        reader = ACCameraPoseShm.__new__(ACCameraPoseShm)
        reader._mm = mm
        reader._packet_offset_votes = __import__("collections").Counter()
        reader._packet_render_offset = None
        reader._last_collision_seq = 0
        ring.collision_write_index = 1
        ring.collision_slot_seq[0] = 2
        ring.collision_render_frame[0] = 100
        ring.collision_sim_time_ms[0] = 1234
        ring.collision_lap_time_ms[0] = 1000
        ring.collision_lap_count[0] = 0
        ring.collision_car_id[0], ring.collision_with[0] = 0, 3
        ring.collision_depth[0] = .08
        ring.collision_pos_forward[0] = 2.1
        ring.collision_relative_speed_mps[0] = 7.5
        for zone in range(5):
            ring.collision_damage[zone] = zone + 1

        first = reader.read(1000, 0)
        second = reader.read(1000, 0)
        self.assertEqual(len(first["_collision_events"]), 1)
        self.assertEqual(first["_collision_events"][0]["collision_with"], 3)
        self.assertAlmostEqual(
            first["_collision_events"][0]["collision_relative_speed_mps"], 7.5)
        self.assertEqual(second["_collision_events"], [])
        reader.close()
        del ring
        mm.close()

    def test_packet_relation_overrides_adjacent_render_with_same_game_time(self):
        size = ctypes.sizeof(CameraPoseRing)
        mm = mmap.mmap(-1, size)
        ring = CameraPoseRing.from_buffer(mm)
        ring.magic, ring.version = CAMERA_POSE_MAGIC, CAMERA_POSE_VERSION
        ring.capacity, ring.write_index = CAMERA_POSE_CAPACITY, 3
        for i, render_frame in enumerate((100, 101, 102)):
            ring.slot_seq[i] = 2 * (i + 1)
            ring.render_frame[i] = render_frame
            ring.lap_time_ms[i] = 1000 if i == 0 else 1025
            ring.lap_count[i] = 0
            ring.pos_x[i], ring.pos_y[i], ring.pos_z[i] = i, 0, 1
            ring.forward_x[i], ring.forward_y[i], ring.forward_z[i] = 1, 0, 0
            ring.up_x[i], ring.up_y[i], ring.up_z[i] = 0, 0, 1
            ring.fov_deg[i] = 60

        reader = ACCameraPoseShm.__new__(ACCameraPoseShm)
        reader._mm = mm
        reader._packet_offset_votes = __import__("collections").Counter({-61: 10})
        reader._packet_render_offset = -61
        # Both render 101 and 102 have the same game time. Packet 141 identifies 101.
        got = reader.read(1025, 0, graphics_packet=141)
        self.assertEqual(got["cam_render_frame"], 101)
        self.assertEqual(got["cam_packet_render_error"], 0)
        reader.close()
        del ring
        mm.close()

    def test_image_index_keeps_pose_values(self):
        with tempfile.TemporaryDirectory() as td:
            logger = ImageLogger(Path(td))
            pose = {
                "cam_pos_x": 1.0, "cam_pos_y": 2.0, "cam_pos_z": 3.0,
                "cam_forward_x": 1.0, "cam_forward_y": 0.0, "cam_forward_z": 0.0,
                "cam_up_x": 0.0, "cam_up_y": 0.0, "cam_up_z": 1.0,
                "cam_fov_deg": 56.0,
                "_render_cars": [{"car_id": 1, "car_pos_x": 4.0}],
                "_collision_events": [{
                    "collision_event_seq": 2, "collision_with": 1,
                    "collision_depth_m": .05,
                }],
            }
            logger.log(0.0, np.zeros((8, 8, 3), np.uint8), game_time_ms=10,
                       completed_laps=1, camera_pose=pose)
            logger.flush()
            row = pd.read_parquet(Path(td) / "image_index.parquet").iloc[0]
            self.assertEqual(row.cam_pos_x, 1.0)
            self.assertEqual(row.completed_laps, 1)
            car = pd.read_parquet(Path(td) / "render_car_state.parquet").iloc[0]
            self.assertEqual(car.car_id, 1)
            collision = pd.read_parquet(
                Path(td) / "collision_events.parquet").iloc[0]
            self.assertEqual(collision.collision_with, 1)

    def test_builder_remaps_render_car_rows_to_final_frames(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logger = ImageLogger(root)
            for i in range(3):
                pose = {
                    "cam_pos_x": 0.0, "cam_pos_y": 0.0, "cam_pos_z": 1.0,
                    "cam_forward_x": 1.0, "cam_forward_y": 0.0,
                    "cam_forward_z": 0.0, "cam_up_x": 0.0, "cam_up_y": 0.0,
                    "cam_up_z": 1.0, "cam_fov_deg": 60.0,
                    "_render_cars": [{"car_id": 2, "render_frame": 10 + i,
                                      "car_pos_x": float(i)}],
                }
                if i == 1:
                    pose["_collision_events"] = [{
                        "collision_event_seq": 2,
                        "collision_render_frame": 11,
                        "collision_with": 2,
                        "collision_depth_m": .04,
                    }]
                logger.log(i * .025, np.zeros((8, 8, 3), np.uint8),
                           game_time_ms=1000 + i * 25, completed_laps=0,
                           camera_pose=pose)
            logger.flush()
            pd.DataFrame({"t": [0.0, .025, .05],
                          "timestamp_ac": [0.0, .025, .05],
                          "currentTime": [1000, 1025, 1050],
                          "speed": [1.0, 1.0, 1.0]}).to_parquet(
                              root / "telemetry_raw.parquet", index=False)
            build_dataset(root)
            cars = pd.read_parquet(root / "camera" / "render_car_state.parquet")
            self.assertEqual(cars.frame.tolist(), [0, 1, 2])
            self.assertEqual(cars.car_id.tolist(), [2, 2, 2])
            collisions = pd.read_parquet(
                root / "camera" / "collision_events.parquet")
            self.assertEqual(collisions.frame.tolist(), [1])
            self.assertEqual(collisions.collision_with.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
