"""ac_shmem - AC 공유메모리(Graphics 페이지)에서 '지금 화면에 그려진 프레임의 게임시간'을 읽는다.
=========================================================================================

목적(시간동기, 방법 a): 화면 캡처(grab) 순간에 렌더된 프레임의 게임시간을 함께 기록해서,
이미지↔telemetry 를 **벽시계가 아니라 게임시계**로 페어링한다. 벽시계 페어링은 캡처 파이프라인
지연(렌더→present→grab, ~수십 ms, 변동)이 남지만, Graphics 페이지의 프레임 게임시간을 직접
읽으면 그 지연을 ~1 렌더프레임 수준으로 줄이고 실제 지연 분포도 측정할 수 있다.

AC 는 named shared memory("acpmf_graphics" 등)를 열어두고 **외부 프로세스에서도** 같은 이름으로
mmap 하면 읽을 수 있다(SimHub 등 외부 텔레메트리 앱이 쓰는 방식). 구조체 레이아웃은 플러그인의
sim_info.py(SPageFileGraphic)와 동일하게 맞춘다.
  - packetId     : 렌더 프레임마다 증가하는 카운터
  - iCurrentTime : 현재 프레임의 랩타임(ms) = '화면에 그려진 프레임의 게임시간'
"""

from __future__ import annotations

import ctypes
import math
import mmap
from collections import Counter
from ctypes import c_float, c_int32, c_uint32, c_wchar

AC_STATUS = c_int32
AC_SESSION_TYPE = c_int32
AC_FLAG_TYPE = c_int32


class SPageFileGraphic(ctypes.Structure):
    """AC Graphics 공유메모리 레이아웃(sim_info.py 와 동일). iCurrentTime 오프셋이 맞아야 하므로
    앞쪽 필드를 그대로 유지한다."""
    _pack_ = 4
    _fields_ = [
        ("packetId", c_int32),
        ("status", AC_STATUS),
        ("session", AC_SESSION_TYPE),
        ("currentTime", c_wchar * 15),
        ("lastTime", c_wchar * 15),
        ("bestTime", c_wchar * 15),
        ("split", c_wchar * 15),
        ("completedLaps", c_int32),
        ("position", c_int32),
        ("iCurrentTime", c_int32),
        ("iLastTime", c_int32),
        ("iBestTime", c_int32),
        ("sessionTimeLeft", c_float),
        ("distanceTraveled", c_float),
        ("isInPit", c_int32),
        ("currentSectorIndex", c_int32),
        ("lastSectorTime", c_int32),
        ("numberOfLaps", c_int32),
        ("tyreCompound", c_wchar * 33),
        ("replayTimeMultiplier", c_float),
        ("normalizedCarPosition", c_float),
        ("carCoordinates", c_float * 3),
        ("penaltyTime", c_float),
        ("flag", AC_FLAG_TYPE),
        ("idealLineOn", c_int32),
    ]


class ACGraphicsShm:
    """AC Graphics 공유메모리 리더. 열기 실패(게임 미실행 등)해도 조용히 비활성(available=False)."""

    _NAMES = ("acpmf_graphics", "Local\\acpmf_graphics")

    def __init__(self):
        self._mm = None
        # AC graphics packet은 이 환경에서 CSP render frame의 2배로 증가한다.
        # 세션별 절편은 첫 exact-game-time 샘플에서 자동 학습한다.
        self._g = None
        for name in self._NAMES:
            try:
                mm = mmap.mmap(0, ctypes.sizeof(SPageFileGraphic), name)
                self._g = SPageFileGraphic.from_buffer(mm)
                self._mm = mm
                break
            except Exception:
                self._mm = None
                self._g = None
                continue

    @property
    def available(self) -> bool:
        return self._g is not None

    def read(self):
        """(graphics_packet_id, game_time_ms) 반환. 못 읽으면 (None, None)."""
        if self._g is None:
            return (None, None)
        try:
            return (int(self._g.packetId), int(self._g.iCurrentTime))
        except Exception:
            return (None, None)

    def read_full(self):
        """(packet_id, current_lap_time_ms, completed_laps) 반환."""
        if self._g is None:
            return (None, None, None)
        try:
            return (int(self._g.packetId), int(self._g.iCurrentTime),
                    int(self._g.completedLaps))
        except Exception:
            return (None, None, None)

    def close(self):
        # from_buffer 로 만든 구조체가 살아있으면 mmap.close() 가 BufferError. 참조부터 끊는다.
        self._g = None
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None


# CSP Lua 앱(ac_camera_pose_bridge)이 매 렌더 프레임 기록하는 링 버퍼와 동일한 ABI.
# float/int32만 사용해 LuaJIT FFI와 Python ctypes의 정렬 차이를 피한다.
CAMERA_POSE_CAPACITY = 128
RENDER_CAR_MAX = 32
RENDER_CAR_SLOTS = CAMERA_POSE_CAPACITY * RENDER_CAR_MAX
COLLISION_EVENT_CAPACITY = 256
COLLISION_DAMAGE_ZONES = 5
COLLISION_DAMAGE_SLOTS = COLLISION_EVENT_CAPACITY * COLLISION_DAMAGE_ZONES
CAMERA_POSE_MAGIC = 0x41434350  # ASCII "ACCP"
CAMERA_POSE_VERSION = 3


class CameraPoseRing(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("magic", c_uint32),
        ("version", c_uint32),
        ("capacity", c_uint32),
        ("write_index", c_uint32),
        ("slot_seq", c_uint32 * CAMERA_POSE_CAPACITY),
        ("render_frame", c_int32 * CAMERA_POSE_CAPACITY),
        ("sim_time_ms", c_float * CAMERA_POSE_CAPACITY),
        ("lap_time_ms", c_int32 * CAMERA_POSE_CAPACITY),
        ("lap_count", c_int32 * CAMERA_POSE_CAPACITY),
        # 아래 벡터는 dataset world 좌표계: (x,y,z)=(AC z, AC x, AC y).
        ("pos_x", c_float * CAMERA_POSE_CAPACITY),
        ("pos_y", c_float * CAMERA_POSE_CAPACITY),
        ("pos_z", c_float * CAMERA_POSE_CAPACITY),
        ("forward_x", c_float * CAMERA_POSE_CAPACITY),
        ("forward_y", c_float * CAMERA_POSE_CAPACITY),
        ("forward_z", c_float * CAMERA_POSE_CAPACITY),
        ("up_x", c_float * CAMERA_POSE_CAPACITY),
        ("up_y", c_float * CAMERA_POSE_CAPACITY),
        ("up_z", c_float * CAMERA_POSE_CAPACITY),
        ("fov_deg", c_float * CAMERA_POSE_CAPACITY),
        ("camera_mode", c_int32 * CAMERA_POSE_CAPACITY),
        ("drivable_camera_mode", c_int32 * CAMERA_POSE_CAPACITY),
        ("car_camera_index", c_int32 * CAMERA_POSE_CAPACITY),
        ("focused_car", c_int32 * CAMERA_POSE_CAPACITY),
        ("car_count", c_int32 * CAMERA_POSE_CAPACITY),
        ("car_timestamp_ms", c_float * RENDER_CAR_SLOTS),
        ("car_pos_x", c_float * RENDER_CAR_SLOTS),
        ("car_pos_y", c_float * RENDER_CAR_SLOTS),
        ("car_pos_z", c_float * RENDER_CAR_SLOTS),
        ("car_forward_x", c_float * RENDER_CAR_SLOTS),
        ("car_forward_y", c_float * RENDER_CAR_SLOTS),
        ("car_forward_z", c_float * RENDER_CAR_SLOTS),
        ("car_up_x", c_float * RENDER_CAR_SLOTS),
        ("car_up_y", c_float * RENDER_CAR_SLOTS),
        ("car_up_z", c_float * RENDER_CAR_SLOTS),
        # AABB는 차 로컬축: forward=AC z, right=AC x, up=AC y.
        ("car_aabb_center_forward", c_float * RENDER_CAR_SLOTS),
        ("car_aabb_center_right", c_float * RENDER_CAR_SLOTS),
        ("car_aabb_center_up", c_float * RENDER_CAR_SLOTS),
        ("car_aabb_length", c_float * RENDER_CAR_SLOTS),
        ("car_aabb_width", c_float * RENDER_CAR_SLOTS),
        ("car_aabb_height", c_float * RENDER_CAR_SLOTS),
        ("collision_write_index", c_uint32),
        ("collision_slot_seq", c_uint32 * COLLISION_EVENT_CAPACITY),
        ("collision_render_frame", c_int32 * COLLISION_EVENT_CAPACITY),
        ("collision_sim_time_ms", c_float * COLLISION_EVENT_CAPACITY),
        ("collision_lap_time_ms", c_int32 * COLLISION_EVENT_CAPACITY),
        ("collision_lap_count", c_int32 * COLLISION_EVENT_CAPACITY),
        ("collision_car_id", c_int32 * COLLISION_EVENT_CAPACITY),
        ("collision_with", c_int32 * COLLISION_EVENT_CAPACITY),
        ("collision_depth", c_float * COLLISION_EVENT_CAPACITY),
        ("collision_pos_forward", c_float * COLLISION_EVENT_CAPACITY),
        ("collision_pos_right", c_float * COLLISION_EVENT_CAPACITY),
        ("collision_pos_up", c_float * COLLISION_EVENT_CAPACITY),
        ("collision_relative_speed_mps", c_float * COLLISION_EVENT_CAPACITY),
        ("collision_damage", c_float * COLLISION_DAMAGE_SLOTS),
    ]


class ACCameraPoseShm:
    """CSP가 기록한 실제 렌더 카메라 pose 리더.

    이미지나 이미지 특징은 보지 않는다. AC/CSP의 렌더 상태와 Graphics 공유메모리의
    게임시간만으로 캡처 프레임에 가장 가까운 pose를 선택한다.
    """

    _NAMES = ("AcTools.ACDataset.RenderState.v3",
              "Local\\AcTools.ACDataset.RenderState.v3")

    def __init__(self):
        self._mm = None
        # Learn the session-specific graphics-packet/render-frame intercept.
        # A running mode prevents one adjacent-frame sample at startup from
        # locking the whole capture to the wrong render tick.
        self._packet_offset_votes: Counter[int] = Counter()
        self._packet_render_offset: int | None = None
        self._last_collision_seq = 0
        size = ctypes.sizeof(CameraPoseRing)
        for name in self._NAMES:
            try:
                self._mm = mmap.mmap(0, size, name)
                break
            except Exception:
                self._mm = None
        # Ignore collision events that occurred before this collection started.
        if self._mm is not None:
            try:
                initial = CameraPoseRing.from_buffer_copy(self._mm[:size])
                complete = [int(seq) for seq in initial.collision_slot_seq
                            if seq and not (int(seq) & 1)]
                self._last_collision_seq = max(complete, default=0)
            except Exception:
                pass

    @property
    def available(self) -> bool:
        if self._mm is None:
            return False
        try:
            h = CameraPoseRing.from_buffer_copy(self._mm[:ctypes.sizeof(CameraPoseRing)])
            return (h.magic == CAMERA_POSE_MAGIC and h.version == CAMERA_POSE_VERSION
                    and h.capacity == CAMERA_POSE_CAPACITY and h.write_index > 0)
        except Exception:
            return False

    def read(self, game_time_ms: int | None = None,
             completed_laps: int | None = None,
             graphics_packet: int | None = None) -> dict | None:
        """게임시간에 가장 가까운 완결된 pose 샘플을 dict로 반환."""
        if self._mm is None:
            return None
        size = ctypes.sizeof(CameraPoseRing)
        try:
            # writer가 슬롯을 갱신하는 도중일 수 있어 두 번 복사해 seq가 같은 샘플만 쓴다.
            a = CameraPoseRing.from_buffer_copy(self._mm[:size])
            b = CameraPoseRing.from_buffer_copy(self._mm[:size])
        except Exception:
            return None
        if (b.magic != CAMERA_POSE_MAGIC or b.version != CAMERA_POSE_VERSION
                or b.capacity != CAMERA_POSE_CAPACITY):
            return None

        candidates = []
        for i in range(CAMERA_POSE_CAPACITY):
            seq = int(b.slot_seq[i])
            if seq == 0 or seq & 1 or seq != int(a.slot_seq[i]):
                continue
            if completed_laps is not None and int(b.lap_count[i]) != int(completed_laps):
                continue
            delta = (abs(int(b.lap_time_ms[i]) - int(game_time_ms))
                     if game_time_ms is not None else 0)
            # 같은 game time이면 최신 렌더 샘플을 선택한다(정지/일시정지에서도 결정적).
            candidates.append((delta, -int(b.render_frame[i]), i))
        if not candidates:
            return None
        time_best = min(candidates)
        # 정확히 같은 game time은 보통 유일하다. 이를 이용해 세션별 packet↔render 절편을
        # 이미지나 픽셀 없이 한 번 학습한다.
        if graphics_packet is not None and time_best[0] == 0:
            best_i = time_best[2]
            observed = int(graphics_packet) - 2 * int(b.render_frame[best_i])
            self._packet_offset_votes[observed] += 1
            self._packet_render_offset = self._packet_offset_votes.most_common(1)[0][0]

        if self._packet_render_offset is not None and graphics_packet is not None:
            # game time 최소값에서 20ms 이내인 샘플만 허용한 뒤 packet 관계를 최우선한다.
            # 인코더가 늦게 읽어도 링 버퍼에서 정확한 과거 렌더 tick을 고를 수 있다.
            limit = time_best[0] + 20
            nearby = [x for x in candidates if x[0] <= limit]
            delta, _neg_frame, i = min(
                nearby,
                key=lambda x: (abs((int(graphics_packet)
                                    - 2 * int(b.render_frame[x[2]]))
                                   - self._packet_render_offset),
                               x[0], x[1]))
        else:
            delta, _neg_frame, i = time_best
        vals = [b.pos_x[i], b.pos_y[i], b.pos_z[i],
                b.forward_x[i], b.forward_y[i], b.forward_z[i],
                b.up_x[i], b.up_y[i], b.up_z[i], b.fov_deg[i]]
        if not all(math.isfinite(float(v)) for v in vals):
            return None
        result = {
            "csp_bridge_version": int(b.version),
            "cam_pose_seq": int(b.slot_seq[i]),
            "cam_render_frame": int(b.render_frame[i]),
            "cam_sim_time_ms": float(b.sim_time_ms[i]),
            "cam_game_time_ms": int(b.lap_time_ms[i]),
            "cam_lap_count": int(b.lap_count[i]),
            "cam_pose_game_delta_ms": int(delta),
            "cam_packet_render_offset": self._packet_render_offset,
            "cam_packet_render_error": (
                abs((int(graphics_packet) - 2 * int(b.render_frame[i]))
                    - self._packet_render_offset)
                if graphics_packet is not None and self._packet_render_offset is not None
                else None),
            "cam_pos_x": float(b.pos_x[i]), "cam_pos_y": float(b.pos_y[i]),
            "cam_pos_z": float(b.pos_z[i]),
            "cam_forward_x": float(b.forward_x[i]),
            "cam_forward_y": float(b.forward_y[i]),
            "cam_forward_z": float(b.forward_z[i]),
            "cam_up_x": float(b.up_x[i]), "cam_up_y": float(b.up_y[i]),
            "cam_up_z": float(b.up_z[i]),
            "cam_fov_deg": float(b.fov_deg[i]),
            "cam_mode": int(b.camera_mode[i]),
            "cam_drivable_mode": int(b.drivable_camera_mode[i]),
            "cam_car_index": int(b.car_camera_index[i]),
            "cam_focused_car": int(b.focused_car[i]),
        }
        cars = []
        car_count = max(0, min(int(b.car_count[i]), RENDER_CAR_MAX))
        for car_id in range(car_count):
            k = i * RENDER_CAR_MAX + car_id
            car_vals = [b.car_pos_x[k], b.car_pos_y[k], b.car_pos_z[k],
                        b.car_forward_x[k], b.car_forward_y[k], b.car_forward_z[k],
                        b.car_up_x[k], b.car_up_y[k], b.car_up_z[k],
                        b.car_aabb_length[k], b.car_aabb_width[k], b.car_aabb_height[k]]
            if not all(math.isfinite(float(v)) for v in car_vals):
                continue
            if min(float(b.car_aabb_length[k]), float(b.car_aabb_width[k]),
                   float(b.car_aabb_height[k])) <= 0:
                continue
            cars.append({
                "car_id": car_id,
                "render_frame": int(b.render_frame[i]),
                "sim_time_ms": float(b.sim_time_ms[i]),
                "car_timestamp_ms": float(b.car_timestamp_ms[k]),
                "car_state_age_ms": float(b.sim_time_ms[i] - b.car_timestamp_ms[k]),
                "car_pos_x": float(b.car_pos_x[k]), "car_pos_y": float(b.car_pos_y[k]),
                "car_pos_z": float(b.car_pos_z[k]),
                "car_forward_x": float(b.car_forward_x[k]),
                "car_forward_y": float(b.car_forward_y[k]),
                "car_forward_z": float(b.car_forward_z[k]),
                "car_up_x": float(b.car_up_x[k]), "car_up_y": float(b.car_up_y[k]),
                "car_up_z": float(b.car_up_z[k]),
                "aabb_center_forward": float(b.car_aabb_center_forward[k]),
                "aabb_center_right": float(b.car_aabb_center_right[k]),
                "aabb_center_up": float(b.car_aabb_center_up[k]),
                "aabb_length": float(b.car_aabb_length[k]),
                "aabb_width": float(b.car_aabb_width[k]),
                "aabb_height": float(b.car_aabb_height[k]),
            })
        result["_render_cars"] = cars
        collision_events = []
        last_collision_seq = getattr(self, "_last_collision_seq", 0)
        newest_collision_seq = last_collision_seq
        for j in range(COLLISION_EVENT_CAPACITY):
            seq = int(b.collision_slot_seq[j])
            if (seq == 0 or seq & 1 or seq != int(a.collision_slot_seq[j])
                    or seq <= last_collision_seq):
                continue
            values = [
                b.collision_sim_time_ms[j], b.collision_depth[j],
                b.collision_pos_forward[j], b.collision_pos_right[j],
                b.collision_pos_up[j], b.collision_relative_speed_mps[j],
            ]
            damage = [
                float(b.collision_damage[j * COLLISION_DAMAGE_ZONES + zone])
                for zone in range(COLLISION_DAMAGE_ZONES)
            ]
            if not all(math.isfinite(float(value)) for value in values + damage):
                continue
            collision_events.append({
                "collision_event_seq": seq,
                "collision_render_frame": int(b.collision_render_frame[j]),
                "collision_sim_time_ms": float(b.collision_sim_time_ms[j]),
                "collision_lap_time_ms": int(b.collision_lap_time_ms[j]),
                "collision_lap_count": int(b.collision_lap_count[j]),
                "collision_car_id": int(b.collision_car_id[j]),
                "collision_with": int(b.collision_with[j]),
                "collision_depth_m": float(b.collision_depth[j]),
                "collision_position_forward": float(b.collision_pos_forward[j]),
                "collision_position_right": float(b.collision_pos_right[j]),
                "collision_position_up": float(b.collision_pos_up[j]),
                "collision_relative_speed_mps": float(
                    b.collision_relative_speed_mps[j]),
                **{f"collision_damage_{zone}": damage[zone]
                   for zone in range(COLLISION_DAMAGE_ZONES)},
            })
            newest_collision_seq = max(newest_collision_seq, seq)
        if collision_events:
            collision_events.sort(key=lambda event: event["collision_event_seq"])
            self._last_collision_seq = newest_collision_seq
        result["_collision_events"] = collision_events
        return result

    def close(self):
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
