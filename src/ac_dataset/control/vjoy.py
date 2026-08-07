"""Small ctypes wrapper for the installed vJoy 2.x driver."""

from __future__ import annotations

import ctypes
from pathlib import Path
import struct


AXIS_MAX = 0x8000
DEFAULT_DLL = Path(r"C:\Program Files\vJoy\x64\vJoyInterface.dll")
POSITION_FORMAT = "BlllllllllllllllllllIIII"


class VJoyDevice:
    def __init__(self, device_id: int = 1, dll_path: Path = DEFAULT_DLL):
        if not dll_path.exists():
            raise FileNotFoundError(f"vJoy interface DLL not found: {dll_path}")
        self.device_id = int(device_id)
        self.dll = ctypes.WinDLL(str(dll_path))
        self.dll.vJoyEnabled.restype = ctypes.c_bool
        self.dll.GetvJoyVersion.restype = ctypes.c_short
        self.dll.GetVJDStatus.argtypes = [ctypes.c_uint]
        self.dll.GetVJDStatus.restype = ctypes.c_int
        self.dll.AcquireVJD.argtypes = [ctypes.c_uint]
        self.dll.AcquireVJD.restype = ctypes.c_bool
        self.dll.RelinquishVJD.argtypes = [ctypes.c_uint]
        self.dll.UpdateVJD.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        self.dll.UpdateVJD.restype = ctypes.c_bool
        if not self.dll.vJoyEnabled():
            raise RuntimeError("vJoy driver is installed but not enabled")
        if not self.dll.AcquireVJD(self.device_id):
            status = int(self.dll.GetVJDStatus(self.device_id))
            raise RuntimeError(f"could not acquire vJoy device {self.device_id}; status={status}")
        self.acquired = True
        self.set_controls(0.0, 0.0, 0.0)

    @property
    def version(self) -> int:
        return int(self.dll.GetvJoyVersion())

    @staticmethod
    def _axis(value: float, centered: bool = False) -> int:
        if centered:
            value = (max(-1.0, min(1.0, float(value))) + 1.0) * 0.5
        else:
            value = max(0.0, min(1.0, float(value)))
        return int(round(value * AXIS_MAX))

    def set_controls(self, steer: float, throttle: float, brake: float) -> None:
        values = {
            "wAxisX": self._axis(steer, centered=True),
            "wAxisY": self._axis(throttle),
            "wAxisZ": self._axis(brake),
        }
        fields = [0] * 24
        fields[0] = self.device_id
        # B, then throttle/rudder/aileron/X/Y/Z...; X/Y/Z are long indices 4/5/6.
        fields[4], fields[5], fields[6] = values["wAxisX"], values["wAxisY"], values["wAxisZ"]
        packed = struct.pack(POSITION_FORMAT, *fields)
        buffer = ctypes.create_string_buffer(packed)
        if not self.dll.UpdateVJD(self.device_id, ctypes.byref(buffer)):
            raise RuntimeError(f"vJoy UpdateVJD failed for device {self.device_id}")

    def close(self) -> None:
        if not getattr(self, "acquired", False):
            return
        try:
            self.set_controls(0.0, 0.0, 0.0)
        finally:
            self.dll.RelinquishVJD(self.device_id)
            self.acquired = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
