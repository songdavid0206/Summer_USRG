"""AC 데이터셋 취득 파이프라인 - recorder 패키지.

- telemetry_logger : ego + action + opponents raw 스트림 기록 (각자 timestamp)
- image_logger     : 화면 캡처 raw 스트림 기록 (frame별 timestamp)
- sync             : 두 raw 스트림 -> frame 단위 samples 로 병합 (오프라인)
"""

from .telemetry_logger import TelemetryLogger
from .image_logger import ImageLogger
from .ac_telemetry import ACTelemetrySource
from .ac_opponents import ACOpponentsSource
from . import sync

__all__ = ["TelemetryLogger", "ImageLogger", "ACTelemetrySource",
           "ACOpponentsSource", "sync"]
