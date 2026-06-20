#!/usr/bin/env python3
"""Create CSV and Markdown summaries from VOID benchmark JSON reports."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "report",
    "created_at_utc",
    "label",
    "scenario",
    "gpu_memory_mode",
    "gpu_name",
    "num_inference_steps",
    "sample_size",
    "max_video_length",
    "temporal_window_size",
    "run_seqs",
    "repeat_index",
    "returncode",
    "wall_time_s",
    "init_before_transformer_s",
    "load_transformer_to_sequence_s",
    "time_to_first_sequence_log_s",
    "post_first_sequence_log_s",
    "max_memory_used_gb",
]


def parse_args() -> argparse.Namespace:
    default_reports = Path(__file__).resolve().parents[1] / "benchmark" / "reports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path, help="JSON report files. Defaults to benchmark/reports/*.json.")
    parser.add_argument("--reports-dir", type=Path, default=default_reports)
    parser.add_argument("--out-dir", type=Path, default=default_reports)
    parser.add_argument("--label", default="void_benchmark_summary")
    return parser.parse_args()


def load_reports(args: argparse.Namespace) -> list[Path]:
    if args.reports:
        return args.reports
    return sorted(args.reports_dir.glob("*.json"))


def value(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def report_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmark = payload.get("benchmark", {})
    environment = payload.get("environment", {})
    torch_env = environment.get("torch") if isinstance(environment, dict) else {}
    gpu_name = value(torch_env or {}, "device_name") or "unknown"
    rows = []
    for run in payload.get("runs", []):
        durations = value(run, "phases", "durations_s") or {}
        gpu_samples = run.get("gpu_samples") or {}
        max_memory_mb = gpu_samples.get("max_memory_used_mb")
        rows.append(
            {
                "report": str(path),
                "created_at_utc": payload.get("created_at_utc"),
                "label": payload.get("label"),
                "scenario": benchmark.get("scenario"),
                "gpu_memory_mode": benchmark.get("gpu_memory_mode"),
                "gpu_name": gpu_name,
                "num_inference_steps": benchmark.get("num_inference_steps"),
                "sample_size": benchmark.get("sample_size"),
                "max_video_length": benchmark.get("max_video_length"),
                "temporal_window_size": benchmark.get("temporal_window_size"),
                "run_seqs": benchmark.get("run_seqs"),
                "repeat_index": run.get("repeat_index"),
                "returncode": run.get("returncode"),
                "wall_time_s": run.get("wall_time_s"),
                "init_before_transformer_s": durations.get("init_before_transformer_s"),
                "load_transformer_to_sequence_s": durations.get("load_transformer_to_sequence_s"),
                "time_to_first_sequence_log_s": durations.get("time_to_first_sequence_log_s"),
                "post_first_sequence_log_s": durations.get("post_first_sequence_log_s"),
                "max_memory_used_gb": round(max_memory_mb / 1024, 3) if isinstance(max_memory_mb, (int, float)) else None,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "scenario",
        "gpu_memory_mode",
        "gpu_name",
        "repeat_index",
        "wall_time_s",
        "init_before_transformer_s",
        "load_transformer_to_sequence_s",
        "post_first_sequence_log_s",
        "max_memory_used_gb",
        "returncode",
    ]
    lines = ["# VOID Benchmark Summary", ""]
    if not rows:
        lines.append("No benchmark rows found.")
    else:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for report in load_reports(args):
        rows.extend(report_rows(report))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{args.label}"
    csv_path = args.out_dir / f"{base}.csv"
    md_path = args.out_dir / f"{base}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"Wrote CSV summary: {csv_path}")
    print(f"Wrote Markdown summary: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
