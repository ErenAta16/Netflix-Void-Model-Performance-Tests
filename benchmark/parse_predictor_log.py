#!/usr/bin/env python3
"""Parse VOID predictor logs into coarse timing phases.

The upstream predictor does not expose profiling markers for every internal
step, so this parser only derives brackets from observable log lines. The
numbers should be treated as benchmark markers, not exact internal timings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Marker:
    name: str
    needle: str


MARKERS = (
    Marker("first_output", ""),
    Marker("transformer_load_log", "Load transformer from checkpoint"),
    Marker("first_sequence_log", "Sequence to run:"),
    Marker("mask_load_log", "dataloading mask"),
    Marker("denoise_config_log", "Use cfg:"),
)


def parse_phase_markers(log_records: list[dict[str, Any]], wall_time_s: float | None) -> dict[str, Any]:
    """Return phase marker offsets and derived durations.

    Each log record is expected to contain:
    - ``elapsed_s``: seconds since the child process was spawned
    - ``text``: the emitted log line
    """

    marker_offsets: dict[str, float | None] = {marker.name: None for marker in MARKERS}
    for record in log_records:
        elapsed = record.get("elapsed_s")
        text = str(record.get("text", ""))
        if elapsed is None:
            continue
        if marker_offsets["first_output"] is None:
            marker_offsets["first_output"] = float(elapsed)
        for marker in MARKERS:
            if marker.needle and marker_offsets[marker.name] is None and marker.needle in text:
                marker_offsets[marker.name] = float(elapsed)

    first_output = marker_offsets["first_output"]
    transformer_load = marker_offsets["transformer_load_log"]
    first_sequence = marker_offsets["first_sequence_log"]
    mask_load = marker_offsets["mask_load_log"]
    denoise_config = marker_offsets["denoise_config_log"]

    durations = {
        "process_to_first_output_s": delta(0.0, first_output),
        "init_before_transformer_s": delta(first_output, transformer_load),
        "process_to_transformer_load_s": delta(0.0, transformer_load),
        "load_transformer_to_sequence_s": delta(transformer_load, first_sequence),
        "time_to_first_sequence_log_s": first_sequence,
        "sequence_to_mask_load_s": delta(first_sequence, mask_load),
        "sequence_to_denoise_config_s": delta(first_sequence, denoise_config),
        "post_first_sequence_log_s": delta(first_sequence, wall_time_s),
    }

    return {
        "markers_s": marker_offsets,
        "durations_s": durations,
        "notes": [
            "init_before_transformer_s is measured from the first child-process output to the transformer-load log.",
            "load_transformer_to_sequence_s is the bracket between two log lines, not pure safetensors I/O.",
        ],
    }


def delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round(float(end) - float(start), 3)
