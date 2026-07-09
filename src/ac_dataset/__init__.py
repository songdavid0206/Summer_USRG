"""AC 데이터셋 취득 파이프라인 (ac_dataset).

사람이 Assetto Corsa를 직접 주행하는 동안 이미지 + 자차(ego) 상태 + 상대차 + 조작값을
수집해, 카메라 프레임 기준으로 정렬된 dataset(camera/ ego_state/ opponent/)으로 저장한다.

공개 API
--------
- collect        : 주행 중 수집 메인 루프 (게임 필요)
- build_dataset  : raw 스트림 -> camera/ego_state/opponent 폴더로 정렬 (오프라인)
"""

from .collect import collect
from .recorder.dataset_builder import build_dataset
from .track import parse_fast_lane, extract_track_for_run, identify_track

__all__ = ["collect", "build_dataset",
           "parse_fast_lane", "extract_track_for_run", "identify_track"]
__version__ = "0.1.0"
