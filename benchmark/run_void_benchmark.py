#!/usr/bin/env python3
"""Run a repeatable VOID inference benchmark and save machine-readable results.

The wrapper is intentionally thin: it uses Netflix VOID's upstream
predict_v2v.py process, records environment metadata, and derives coarse phase
brackets from observable predictor log lines.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from parse_predictor_log import parse_phase_markers


BASE_MODEL_REPO = "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"
BASE_MODEL_DIR = "CogVideoX-Fun-V1.5-5b-InP"
VOID_MODEL_REPO = "netflix/void-model"
VOID_CHECKPOINT = "void_pass1.safetensors"
REQUIRED_SEQUENCE_FILES = ("input_video.mp4", "quadmask_0.mp4", "prompt.json")


def parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[1]
    default_void_root = Path("/content/void-model") if Path("/content").exists() else default_repo_root / "void-model"
    default_save_path = Path("/content/void_outputs") if Path("/content").exists() else default_repo_root / "void_outputs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument("--void-root", type=Path, default=default_void_root)
    parser.add_argument("--setup-void", action="store_true", help="Clone/update Netflix VOID and install requirements.")
    parser.add_argument("--download-models", action="store_true", help="Download CogVideoX base model and VOID checkpoint.")
    parser.add_argument("--prepare-sample-data", action="store_true", help="Copy this repo's sample files into VOID custom_data.")
    parser.add_argument(
        "--no-clean-seq-dir",
        action="store_false",
        dest="clean_seq_dir",
        help="Do not delete the target sequence directory before copying sample files.",
    )
    parser.add_argument("--sample-root", type=Path, default=default_repo_root)
    parser.add_argument("--seq-name", default="my_video", help="Sequence folder name to prepare from sample files.")
    parser.add_argument("--run-seqs", default="", help="Comma-separated sequence names. Defaults to --seq-name.")
    parser.add_argument(
        "--scenario",
        choices=("cold_single_seq", "batched_multi_seq"),
        default="cold_single_seq",
        help="Benchmark one cold sequence for latency or all prepared sequences for amortized throughput.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="VOID data root. Defaults to <void-root>/custom_data.")
    parser.add_argument("--save-path", type=Path, default=default_save_path)
    parser.add_argument("--reports-dir", type=Path, default=default_repo_root / "benchmark" / "reports")
    parser.add_argument("--label", default="void_benchmark")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0, help="Seconds between nvidia-smi samples. Use 0 to disable.")
    parser.add_argument(
        "--gpu-memory-mode",
        default="model_cpu_offload_and_qfloat8",
        choices=("model_cpu_offload_and_qfloat8", "model_full_load", "model_cpu_offload", "sequential_cpu_offload"),
    )
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--sample-size", default="384x672")
    parser.add_argument("--max-video-length", type=int, default=197)
    parser.add_argument("--temporal-window-size", type=int, default=85)
    parser.add_argument("--python-importtime", action="store_true", help="Run predictor with python -X importtime.")
    parser.add_argument("--extra-config", action="append", default=[], help="Extra ml_collections override, e.g. --config.data.max_video_length=85")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def git_value(repo_root: Path, *args: str) -> str | None:
    try:
        result = run(["git", *args], cwd=repo_root, capture=True)
        return (result.stdout or "").strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def setup_void(void_root: Path) -> None:
    if (void_root / ".git").exists():
        run(["git", "-C", str(void_root), "pull", "--ff-only"])
    else:
        void_root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "https://github.com/Netflix/void-model.git", str(void_root)])

    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=void_root)


def download_models(void_root: Path) -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(
        repo_id=BASE_MODEL_REPO,
        local_dir=str(void_root / BASE_MODEL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    hf_hub_download(
        repo_id=VOID_MODEL_REPO,
        filename=VOID_CHECKPOINT,
        local_dir=str(void_root),
        local_dir_use_symlinks=False,
    )


def prepare_sample_data(sample_root: Path, data_root: Path, seq_name: str, clean_seq_dir: bool = True) -> Path:
    missing = [name for name in REQUIRED_SEQUENCE_FILES if not (sample_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sample files under {sample_root}: {', '.join(missing)}")

    seq_dir = data_root / seq_name
    if clean_seq_dir and seq_dir.exists():
        shutil.rmtree(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_SEQUENCE_FILES:
        shutil.copy2(sample_root / name, seq_dir / name)
    return seq_dir


def discover_prepared_sequences(data_root: Path) -> list[str]:
    if not data_root.exists():
        return []
    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and all((path / name).exists() for name in REQUIRED_SEQUENCE_FILES)
    )


def inspect_sequence_inputs(data_root: Path, sequence_names: list[str]) -> dict[str, Any]:
    inspection: dict[str, Any] = {}
    for name in sequence_names:
        seq_dir = data_root / name
        quadmasks = sorted(path.name for path in seq_dir.glob("quadmask_*.mp4")) if seq_dir.exists() else []
        masks = sorted(path.name for path in seq_dir.glob("mask_*.mp4")) if seq_dir.exists() else []
        inspection[name] = {
            "exists": seq_dir.exists(),
            "quadmask_files": quadmasks,
            "mask_files": masks,
            "required_files_present": all((seq_dir / required).exists() for required in REQUIRED_SEQUENCE_FILES),
        }
    return inspection


def inspect_model_assets(void_root: Path) -> dict[str, Any]:
    base_model = void_root / BASE_MODEL_DIR
    checkpoint = void_root / VOID_CHECKPOINT
    return {
        "base_model_dir_exists": base_model.exists(),
        "base_transformer_config_exists": (base_model / "transformer" / "config.json").exists(),
        "void_checkpoint_exists": checkpoint.exists(),
        "void_checkpoint_size_gb": round(checkpoint.stat().st_size / 1e9, 3) if checkpoint.exists() else None,
    }


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "package_versions": {
            name: package_version(name)
            for name in ("torch", "diffusers", "transformers", "accelerate", "optimum-quanto")
        },
    }

    try:
        import torch

        torch_info: dict[str, Any] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
        }
        if torch.cuda.is_available():
            torch_info.update(
                {
                    "device_count": torch.cuda.device_count(),
                    "device_name": torch.cuda.get_device_name(0),
                    "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3),
                }
            )
        env["torch"] = torch_info
    except Exception as exc:
        env["torch_error"] = repr(exc)

    try:
        result = run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture=True,
        )
        env["nvidia_smi"] = [
            {"name": row[0], "memory_total_mb": row[1], "driver_version": row[2]}
            for row in (line.split(", ") for line in (result.stdout or "").strip().splitlines())
            if len(row) == 3
        ]
    except Exception as exc:
        env["nvidia_smi_error"] = repr(exc)

    return env


def poll_gpu(stop: threading.Event, interval: float, samples: list[dict[str, Any]]) -> None:
    query = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    while not stop.is_set():
        try:
            result = subprocess.run(query, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            for line in result.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 4:
                    samples.append(
                        {
                            "timestamp": parts[0],
                            "name": parts[1],
                            "memory_used_mb": int(float(parts[2])),
                            "utilization_gpu_pct": int(float(parts[3])),
                        }
                    )
        except Exception:
            pass
        stop.wait(interval)


def summarize_gpu_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}
    return {
        "sample_count": len(samples),
        "min_memory_used_mb": min(sample["memory_used_mb"] for sample in samples),
        "max_memory_used_mb": max(sample["memory_used_mb"] for sample in samples),
        "last_memory_used_mb": samples[-1]["memory_used_mb"],
        "max_utilization_gpu_pct": max(sample["utilization_gpu_pct"] for sample in samples),
    }


def stream_predictor(cmd: list[str], cwd: Path, gpu_sample_interval: float) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    line_tail: list[str] = []
    log_records: list[dict[str, Any]] = []
    first_sequence_at: float | None = None
    gpu_samples: list[dict[str, Any]] = []
    stop = threading.Event()
    sampler: threading.Thread | None = None

    if gpu_sample_interval > 0:
        sampler = threading.Thread(target=poll_gpu, args=(stop, gpu_sample_interval, gpu_samples), daemon=True)
        sampler.start()

    started = time.perf_counter()
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        elapsed_s = round(time.perf_counter() - started, 3)
        print(line, end="", flush=True)
        text = line.rstrip()
        line_tail.append(text)
        log_records.append({"elapsed_s": elapsed_s, "text": text})
        if len(line_tail) > 200:
            line_tail.pop(0)
        if first_sequence_at is None and "Sequence to run:" in line:
            first_sequence_at = time.perf_counter()
    return_code = proc.wait()
    finished = time.perf_counter()

    stop.set()
    if sampler is not None:
        sampler.join(timeout=2)

    wall_time_s = round(finished - started, 3)
    time_to_first_sequence_log_s = round(first_sequence_at - started, 3) if first_sequence_at else None
    post_first_sequence_log_s = round(finished - first_sequence_at, 3) if first_sequence_at else None
    phases = parse_phase_markers(log_records, wall_time_s)

    return {
        "returncode": return_code,
        "wall_time_s": wall_time_s,
        "time_to_first_sequence_log_s": time_to_first_sequence_log_s,
        "post_first_sequence_log_s": post_first_sequence_log_s,
        "phases": phases,
        "gpu_samples": summarize_gpu_samples(gpu_samples),
        "log_record_tail": log_records[-200:],
        "log_tail": line_tail,
    }


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "void_benchmark"


def seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def build_config_overrides(args: argparse.Namespace) -> list[str]:
    return [
        f"--config.system.gpu_memory_mode={args.gpu_memory_mode}",
        "--config.experiment.skip_if_exists=False",
        f"--config.video_model.num_inference_steps={int(args.num_inference_steps)}",
        f"--config.data.sample_size={args.sample_size}",
        f"--config.data.max_video_length={int(args.max_video_length)}",
        f"--config.video_model.temporal_window_size={int(args.temporal_window_size)}",
        *args.extra_config,
    ]


def predictor_python(args: argparse.Namespace) -> list[str]:
    if args.python_importtime:
        return [sys.executable, "-X", "importtime"]
    return [sys.executable]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    void_root = args.void_root.resolve()
    data_root = (args.data_root or (void_root / "custom_data")).resolve()
    save_path = args.save_path.resolve()
    model_assets_before_download = inspect_model_assets(void_root)

    if args.setup_void:
        setup_void(void_root)
    if args.download_models:
        download_models(void_root)
    if args.prepare_sample_data:
        prepared = prepare_sample_data(args.sample_root.resolve(), data_root, args.seq_name, args.clean_seq_dir)
        print(f"Prepared sequence data: {prepared}")

    prepared_sequences = discover_prepared_sequences(data_root)
    if args.run_seqs:
        run_seqs = args.run_seqs
    elif args.scenario == "batched_multi_seq":
        run_seqs = ",".join(prepared_sequences)
    else:
        run_seqs = args.seq_name
    if not run_seqs:
        raise ValueError("No sequences selected. Prepare sample data or pass --run-seqs.")
    selected_sequences = [seq for seq in run_seqs.split(",") if seq]

    if not (void_root / "inference" / "cogvideox_fun" / "predict_v2v.py").exists():
        raise FileNotFoundError(f"VOID predictor not found under {void_root}. Use --setup-void first.")
    if not (void_root / BASE_MODEL_DIR / "transformer" / "config.json").exists():
        raise FileNotFoundError(f"Base model is missing under {void_root / BASE_MODEL_DIR}. Use --download-models first.")
    if not (void_root / VOID_CHECKPOINT).exists():
        raise FileNotFoundError(f"VOID checkpoint is missing under {void_root}. Use --download-models first.")

    model_assets_after_download = inspect_model_assets(void_root)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)
    environment = collect_environment()
    config_overrides = build_config_overrides(args)

    gpu_name = "unknown_gpu"
    torch_env = environment.get("torch")
    if isinstance(torch_env, dict):
        gpu_name = torch_env.get("device_name") or gpu_name
    filename = (
        dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        + "_"
        + sanitize_label(args.label)
        + "_"
        + sanitize_label(args.scenario)
        + "_"
        + sanitize_label(args.gpu_memory_mode)
        + "_"
        + sanitize_label(gpu_name)
        + ".json"
    )
    report_path = args.reports_dir / filename
    report_stem = Path(filename).stem

    report: dict[str, Any] = {
        "schema_version": 2,
        "label": args.label,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "repo": {
            "root": str(repo_root),
            "branch": git_value(repo_root, "branch", "--show-current"),
            "commit": git_value(repo_root, "rev-parse", "HEAD"),
        },
        "void_repo": {
            "root": str(void_root),
            "branch": git_value(void_root, "branch", "--show-current") if (void_root / ".git").exists() else None,
            "commit": git_value(void_root, "rev-parse", "HEAD") if (void_root / ".git").exists() else None,
        },
        "void_root": str(void_root),
        "environment": environment,
        "model_assets": {
            "before_download": model_assets_before_download,
            "after_download": model_assets_after_download,
        },
        "benchmark": {
            "scenario": args.scenario,
            "run_seqs": run_seqs,
            "prepared_sequences": prepared_sequences,
            "selected_sequence_inputs": inspect_sequence_inputs(data_root, selected_sequences),
            "data_root": str(data_root),
            "save_path": str(save_path),
            "repeats": args.repeats,
            "clean_seq_dir": args.clean_seq_dir,
            "gpu_memory_mode": args.gpu_memory_mode,
            "num_inference_steps": args.num_inference_steps,
            "sample_size": args.sample_size,
            "max_video_length": args.max_video_length,
            "temporal_window_size": args.temporal_window_size,
            "python_importtime": args.python_importtime,
            "config_overrides": config_overrides,
            "extra_config": args.extra_config,
        },
        "runs": [],
    }
    write_report(report_path, report)

    for index in range(args.repeats):
        run_save_path = save_path / f"{report_stem}_repeat_{index + 1:02d}"
        run_save_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            *predictor_python(args),
            "inference/cogvideox_fun/predict_v2v.py",
            "--config",
            "config/quadmask_cogvideox.py",
            f"--config.data.data_rootdir={data_root}",
            f"--config.experiment.run_seqs={run_seqs}",
            f"--config.experiment.save_path={run_save_path}",
            f"--config.video_model.transformer_path={void_root / VOID_CHECKPOINT}",
            *config_overrides,
        ]
        result = stream_predictor(cmd, cwd=void_root, gpu_sample_interval=args.gpu_sample_interval)
        result["repeat_index"] = index + 1
        result["save_path"] = str(run_save_path)
        result["command"] = cmd
        report["runs"].append(result)
        write_report(report_path, report)
        durations = result["phases"]["durations_s"]
        print(
            "Run summary: "
            f"wall={seconds(result['wall_time_s'])}, "
            f"init_before_transformer={seconds(durations['init_before_transformer_s'])}, "
            f"load_transformer_to_sequence={seconds(durations['load_transformer_to_sequence_s'])}, "
            f"after_first_sequence_log={seconds(durations['post_first_sequence_log_s'])}",
            flush=True,
        )
        if result["returncode"] != 0:
            break

    write_report(report_path, report)
    print(f"\nBenchmark report written to: {report_path}")

    failed = [run for run in report["runs"] if run["returncode"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
