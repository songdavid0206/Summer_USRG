"""Assetto Corsa dataset collection and export package."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "collect", "build_dataset", "parse_fast_lane", "extract_track_for_run",
    "identify_track", "build_processed", "export_run", "export_structured",
    "validate_run",
]
__version__ = "0.2.0"

_LAZY_EXPORTS = {
    "collect": (".collect", "collect"),
    "build_dataset": (".recorder.dataset_builder", "build_dataset"),
    "parse_fast_lane": (".track", "parse_fast_lane"),
    "extract_track_for_run": (".track", "extract_track_for_run"),
    "identify_track": (".track", "identify_track"),
    "build_processed": (".processed_builder", "build_processed"),
    "export_run": (".export_frames", "export_run"),
    "export_structured": (".structured_export", "export_structured"),
    "validate_run": (".validate", "validate_run"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
