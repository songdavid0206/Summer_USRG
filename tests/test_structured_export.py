import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ac_dataset.structured_export import (
    _clean_multi_index, _complete_laps, _contact_interactions,
    _export_track_info, _interaction_group, _interaction_partition,
    _interaction_records, _interaction_union_index, _link_selection,
    _non_interaction_segments, _reset_clock_coherent, _session_name,
    _vehicle_name, _weather_folder_name,
    Interaction, detect_interactions,
)


class TrackInfoExportTest(unittest.TestCase):
    def test_exports_racing_line_and_explicit_boundaries(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            session = Path(tmp) / "session"
            track = run / "track"
            track.mkdir(parents=True)
            pd.DataFrame({
                "i": [0, 1, 2, 3],
                "x": [0.0, 10.0, 10.0, 0.0],
                "y": [0.0, 0.0, 10.0, 10.0],
                "z": [1.0, 1.0, 1.0, 1.0],
                "dist": [0.0, 10.0, 20.0, 30.0],
                "side_left": [5.0] * 4,
                "side_right": [4.0] * 4,
                "ai_speed_kmh": [100.0] * 4,
            }).to_parquet(track / "racing_line.parquet", index=False)
            (track / "track.json").write_text(
                '{"track_name":"test_track","coord_frame":"test xyz"}',
                encoding="utf-8")

            info = _export_track_info(run, session)

            self.assertTrue(info["available"])
            self.assertEqual(info["track_name"], "test_track")
            self.assertTrue((session / "TRACK_INFO" / "AI_RACING_LINE.csv").exists())
            self.assertTrue((session / "TRACK_INFO" / "LEFT_BOUNDARY.csv").exists())
            self.assertTrue((session / "TRACK_INFO" / "RIGHT_BOUNDARY.csv").exists())
            exported = pd.read_csv(
                session / "TRACK_INFO" / "AI_RACING_LINE.csv")
            self.assertIn("normalized_position", exported.columns)


class InteractionDetectionTest(unittest.TestCase):
    def test_overtake_has_opponent_identity(self):
        t = np.arange(0.0, 8.0, 0.025)
        forward = np.linspace(12.0, -12.0, len(t))
        table = pd.DataFrame({
            "t_img": t,
            "opp_forward": forward,
            "opp_lateral": np.full(len(t), 2.0),
            "distance": np.hypot(forward, 2.0),
            "valid": np.ones(len(t), dtype=bool),
        })
        events = detect_interactions({3: table})
        overtakes = [event for event in events if event.interaction_type == "overtake"]
        self.assertEqual(len(overtakes), 1)
        self.assertEqual(overtakes[0].opponent_id, 3)
        self.assertTrue(overtakes[0].success)

    def test_distance_alone_is_not_labeled_collision(self):
        t = np.arange(0.0, 2.0, 0.025)
        table = pd.DataFrame({
            "t_img": t,
            "opp_forward": np.zeros(len(t)),
            "opp_lateral": np.full(len(t), 0.5),
            "distance": np.full(len(t), 0.5),
            "valid": np.ones(len(t), dtype=bool),
        })
        events = detect_interactions({1: table})
        self.assertNotIn("collision", {event.interaction_type for event in events})

    def test_center_crossing_without_body_clearance_is_attempt(self):
        t = np.arange(0.0, 8.0, 0.025)
        forward = np.linspace(12.0, -12.0, len(t))
        table = pd.DataFrame({
            "frame": np.arange(len(t)),
            "t_img": t,
            "opp_forward": forward,
            "opp_lateral": np.full(len(t), 2.0),
            "distance": np.hypot(forward, 2.0),
            "valid": np.ones(len(t), dtype=bool),
        })
        never_clear = np.full(len(t), -0.1)
        events = detect_interactions({3: table}, {3: (never_clear, never_clear)})
        attempts = [event for event in events if event.interaction_type == "overtake_attempt"]
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0].success)

    def test_slow_pass_with_late_body_clearance_is_overtake(self):
        t = np.arange(0.0, 10.0, 0.025)
        forward = np.interp(t, [0.0, 2.0, 3.0, 10.0],
                            [3.0, 0.2, -0.3, -10.0])
        table = pd.DataFrame({
            "frame": np.arange(len(t)),
            "t_img": t,
            "opp_forward": forward,
            "opp_lateral": np.full(len(t), 2.0),
            "distance": np.hypot(forward, 2.0),
            "valid": np.ones(len(t), dtype=bool),
        })
        late_clearance = np.where(t >= 6.0, 1.0, -1.0)

        events = detect_interactions(
            {3: table}, {3: (late_clearance, -late_clearance)})

        overtakes = [
            event for event in events if event.interaction_type == "overtake"]
        self.assertEqual(len(overtakes), 1)
        self.assertTrue(overtakes[0].success)
        self.assertGreaterEqual(overtakes[0].event_time, 6.0)

    def test_body_frame_crossing_without_track_order_change_is_not_overtake(self):
        t = np.arange(0.0, 8.0, 0.025)
        forward = np.linspace(12.0, -12.0, len(t))
        table = pd.DataFrame({
            "t_img": t,
            "opp_forward": forward,
            "opp_lateral": np.full(len(t), 2.0),
            "distance": np.hypot(forward, 2.0),
            "valid": np.ones(len(t), dtype=bool),
        })
        track_order_never_changes = np.full(len(t), 5.0)

        events = detect_interactions(
            {3: table}, track_order_deltas={3: track_order_never_changes})

        self.assertFalse(any(
            event.interaction_type.startswith("overtake") for event in events))

    def test_reverse_pass_is_named_overtaken_by(self):
        t = np.arange(0.0, 8.0, 0.025)
        forward = np.linspace(-12.0, 12.0, len(t))
        table = pd.DataFrame({
            "t_img": t,
            "opp_forward": forward,
            "opp_lateral": np.full(len(t), 2.0),
            "distance": np.hypot(forward, 2.0),
            "valid": np.ones(len(t), dtype=bool),
        })
        events = detect_interactions({4: table})
        labels = {event.interaction_type for event in events}
        self.assertIn("overtaken_by", labels)
        self.assertNotIn("overtaken", labels)

    def test_counter_pass_inside_cooldown_is_retained(self):
        t = np.arange(0.0, 8.0, 0.025)
        forward = np.interp(
            t,
            [0.0, 1.5, 2.5, 3.5, 4.5, 8.0],
            [8.0, 8.0, -8.0, -8.0, 8.0, 8.0],
        )
        table = pd.DataFrame({
            "t_img": t,
            "opp_forward": forward,
            "opp_lateral": np.full(len(t), 2.0),
            "distance": np.hypot(forward, 2.0),
            "valid": np.ones(len(t), dtype=bool),
        })

        events = detect_interactions({3: table})
        pass_labels = [
            event.interaction_type for event in events
            if event.interaction_type in {"overtake", "overtaken_by"}
        ]

        self.assertEqual(pass_labels, ["overtake", "overtaken_by"])

    def test_contact_gt_is_split_from_interval_states(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            path = run / "camera"
            path.mkdir()
            pd.DataFrame([{
                "frame": 10, "t_img": 1.25, "collision_event_seq": 2,
                "collision_with": 3, "collision_depth_m": .07,
                "collision_position_forward": 2.0,
                "collision_position_right": -.5,
                "collision_position_up": .2,
                "collision_relative_speed_mps": 6.5,
                **{f"collision_damage_{i}": float(i) for i in range(5)},
            }]).to_parquet(path / "collision_events.parquet", index=False)
            opponents = {2: pd.DataFrame({
                "t_img": [1.25],
                "distance": [4.9],
            })}
            event = _contact_interactions(run, opponents)[0]
            self.assertEqual(event.interaction_type, "car_contact")
            self.assertEqual(event.opponent_id, 2)
            self.assertEqual(event.collision_with_raw, 3)
            self.assertTrue(event.contact_mapping_valid)
            self.assertAlmostEqual(event.minimum_distance_m, 4.9)
            self.assertEqual(_interaction_group(event), "EVENTS")
            self.assertAlmostEqual(event.collision_relative_speed_mps, 6.5)

    def test_repeated_contact_callbacks_are_clustered_by_opponent(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            path = run / "camera"
            path.mkdir()
            pd.DataFrame([
                {"t_img": 1.00, "collision_event_seq": 2, "collision_with": 3},
                {"t_img": 1.10, "collision_event_seq": 4, "collision_with": 3},
                {"t_img": 1.20, "collision_event_seq": 6, "collision_with": 4},
                {"t_img": 1.40, "collision_event_seq": 8, "collision_with": 3},
                {"t_img": 2.00, "collision_event_seq": 10, "collision_with": 3},
            ]).to_parquet(path / "collision_events.parquet", index=False)

            events = _contact_interactions(run, {})

            self.assertEqual(len(events), 3)
            self.assertEqual(
                [(event.opponent_id, event.collision_event_seq) for event in events],
                [(2, 2), (3, 6), (2, 10)],
            )

    def test_low_track_surface_callback_is_not_barrier_contact(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            path = run / "camera"
            path.mkdir()
            pd.DataFrame([{
                "t_img": 1.0,
                "collision_event_seq": 2,
                "collision_with": 0,
                "collision_position_up": .10,
            }]).to_parquet(path / "collision_events.parquet", index=False)
            self.assertEqual(_contact_interactions(run, {}), [])

    def test_high_track_callback_is_barrier_contact(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            path = run / "camera"
            path.mkdir()
            pd.DataFrame([{
                "t_img": 1.0,
                "collision_event_seq": 2,
                "collision_with": 0,
                "collision_position_up": .30,
            }]).to_parquet(path / "collision_events.parquet", index=False)
            events = _contact_interactions(run, {})
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].interaction_type, "barrier_contact")


class LapSplitTest(unittest.TestCase):
    def test_large_initial_lap_clock_is_valid_for_standing_start(self):
        self.assertTrue(_reset_clock_coherent(
            .005, 18000.0, 18000.0, 0, 0, standing_start_lap=0))

    def test_large_lap_clock_is_rejected_at_later_boundary(self):
        self.assertFalse(_reset_clock_coherent(
            .005, 70000.0, 70000.0, 3, 3, standing_start_lap=0))

    def test_standing_start_keeps_exactly_one_frame_before_motion(self):
        rows = []
        for i, progress in enumerate(np.linspace(.10, .99, 40)):
            rows.append({
                "lap_count": 0,
                "t_img": i * .025,
                "time_ms": i * 25,
                "spline_position": progress,
                "sync_ok": True,
            })
        for i, progress in enumerate(np.linspace(.01, .99, 40), start=40):
            rows.append({
                "lap_count": 1,
                "t_img": i * .025,
                "time_ms": i * 25,
                "spline_position": progress,
                "sync_ok": True,
            })
        laps = _complete_laps(pd.DataFrame(rows), standing_start_time=.20)
        self.assertEqual(len(laps), 2)
        standing = laps[0][1]
        self.assertAlmostEqual(float(standing["t_img"].iloc[0]), .175)
        self.assertEqual(laps[0][0], 0)

    def test_solo_link_renames_to_normalized_time_and_omits_objects(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "flat"
            for subdir, suffix in (
                    ("CAMERA", ".jpg"), ("EGO_INFO", ".txt"),
                    ("OBJECT_INFO", ".txt")):
                source = flat / subdir / f"66125{suffix}"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"frame")
            index = pd.DataFrame({
                "source_time_ms": [66125],
                "time_ms": [0],
            })
            target = root / "lap_01"
            _link_selection(flat, target, index, include_objects=False)
            self.assertTrue((target / "CAMERA" / "0.jpg").exists())
            self.assertTrue((target / "EGO_INFO" / "0.txt").exists())
            self.assertFalse((target / "OBJECT_INFO").exists())

    def test_only_complete_laps_are_numbered(self):
        rows = []
        for lap, progress in (
            (0, np.linspace(0.5, 0.99, 20)),
            (1, np.linspace(0.01, 0.99, 40)),
            (2, np.linspace(0.01, 0.40, 20)),
        ):
            for value in progress:
                rows.append({"lap_count": lap, "t_img": len(rows) * 0.025,
                             "spline_position": value})
        laps = _complete_laps(pd.DataFrame(rows))
        self.assertEqual([source_lap for source_lap, _ in laps], [1])

    def test_lap_count_lagging_one_reset_tick_is_trimmed(self):
        partial = np.linspace(.5, .999, 20)
        complete = np.r_[np.linspace(.01, .999, 40), .0001]
        index = pd.DataFrame({
            "lap_count": np.r_[np.zeros(len(partial), dtype=int),
                               np.ones(len(complete), dtype=int)],
            "t_img": np.arange(len(partial) + len(complete)) * .025,
            "spline_position": np.r_[partial, complete],
        })
        laps = _complete_laps(index)
        self.assertEqual(len(laps), 1)
        self.assertGreater(laps[0][1]["spline_position"].iloc[-1], .99)

    def test_incoherent_lap_clock_boundary_frame_is_not_exported(self):
        index = pd.DataFrame({
            "lap_count": [0, 0, 1, 1, 1, 1],
            "t_img": np.arange(6) * .025,
            "spline_position": [.8, .99, .001, .01, .5, .99],
            "sync_ok": [True, True, False, True, True, True],
        })
        laps = _complete_laps(index)
        self.assertEqual(len(laps), 1)
        self.assertEqual(laps[0][1].index.tolist(), [3, 4, 5])


class SessionNamingTest(unittest.TestCase):
    def test_name_contains_time_track_vehicle_mode_and_rate(self):
        meta = {
            "weather_name": "3_clear",
            "race_cars": [
            {"car_id": 0, "model": "ks_ferrari_sf15t"},
            {"car_id": 1, "model": "other", "ai_level": "95",
             "ai_aggression": "90"},
            {"car_id": 2, "model": "other", "ai_level": "85",
             "ai_aggression": "70"},
        ]}
        vehicle = _vehicle_name(meta)
        self.assertEqual(vehicle, "ferrarisf15t")
        self.assertEqual(
            _session_name("20260723_144221", "ks_laguna_seca",
                          vehicle, "multi", 40, meta),
            "20260723_144221_lagunaseca_ferrarisf15t_clear_multi_40hz_90_80",
        )

    def test_weather_names_are_normalized(self):
        self.assertEqual(_weather_folder_name("5_light_clouds"), "scatteredclouds")
        self.assertEqual(_weather_folder_name("Scattered Clouds"), "scatteredclouds")
        self.assertEqual(_weather_folder_name("mist"), "mist")


class InteractionPartitionTest(unittest.TestCase):
    def test_overlapping_clips_form_unique_union_and_exact_complement(self):
        index = pd.DataFrame({
            "time_ms": np.arange(11) * 1000,
            "t_img": np.arange(11, dtype=float),
        })
        interactions = [
            Interaction("side_by_side", 1, 4.0, 3.0, 5.0, 2.0),
            Interaction("close_follow", 1, 5.0, 4.0, 6.0, 2.0),
        ]

        with_interaction, without_interaction = _interaction_partition(
            index, interactions, 1.0, 1.0)

        self.assertEqual(
            with_interaction["time_ms"].tolist(), [3000, 4000, 5000, 6000])
        self.assertEqual(
            sorted(with_interaction["time_ms"].tolist()
                   + without_interaction["time_ms"].tolist()),
            index["time_ms"].tolist())
        self.assertTrue(
            set(with_interaction["time_ms"]).isdisjoint(
                set(without_interaction["time_ms"])))

    def test_union_index_preserves_unique_multilabel_provenance(self):
        index = pd.DataFrame({
            "source_frame": np.arange(7),
            "time_ms": np.arange(7) * 1000,
            "t_img": np.arange(7, dtype=float),
        })
        interactions = [
            Interaction("side_by_side", 1, 3.0, 2.0, 4.0, 2.0),
            Interaction("close_follow", 1, 4.0, 3.0, 5.0, 2.0),
        ]
        records = _interaction_records(interactions)
        union = _interaction_union_index(index, records, 1.0, 1.0)
        self.assertEqual(union["time_ms"].tolist(), [2000, 3000, 4000, 5000])
        overlap = union[union["time_ms"] == 3000].iloc[0]
        self.assertEqual(
            overlap["interaction_types"], "close_follow,side_by_side")
        self.assertIn("close_follow_0001", overlap["interaction_ids"])
        self.assertIn("side_by_side_0001", overlap["interaction_ids"])

    def test_non_interaction_is_split_at_removed_windows_and_capture_gaps(self):
        complement = pd.DataFrame({
            "time_ms": [0, 25, 100, 125, 200],
            "t_img": [0.0, .025, .100, .125, .200],
        })
        segments = _non_interaction_segments(complement, 40.0)
        self.assertEqual(
            [segment["time_ms"].tolist() for segment in segments],
            [[0, 25], [100, 125], [200]])


class MultiCleaningTest(unittest.TestCase):
    def test_trims_to_last_good_pre_motion_frame_and_records_reasons(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            pd.DataFrame({
                "frame": np.arange(6),
                "cam_pose_seq": [1, 2, 3, 4, np.nan, 6],
                "cam_pos_x": [0.0] * 6,
                "cam_pose_game_delta_ms": [0, 0, 0, 0, 0, 80],
            }).to_parquet(run / "image_index.parquet", index=False)
            index = pd.DataFrame({
                "time_ms": np.arange(6) * 25,
                "source_index": np.arange(6),
                "source_frame": np.arange(6),
                "t_img": np.arange(6) * .025,
                "sync_ok": [True, True, True, True, True, True],
            })
            clean, excluded = _clean_multi_index(
                run, index, {
                    "start_mode": "standing",
                    "standing_start_motion_time_s": .075,
                })
            self.assertEqual(clean["source_frame"].tolist(), [2, 3])
            self.assertEqual(clean["time_ms"].tolist(), [0, 25])
            self.assertEqual(clean["source_time_ms"].tolist(), [50, 75])
            reasons = dict(zip(
                excluded["source_frame"].astype(int), excluded["reasons"]))
            self.assertIn("pre_motion_desktop", reasons[0])
            self.assertIn("camera_pose_missing", reasons[4])
            self.assertIn("pose_delta_gt_50ms", reasons[5])


if __name__ == "__main__":
    unittest.main()
