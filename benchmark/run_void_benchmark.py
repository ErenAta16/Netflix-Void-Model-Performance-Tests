#!/usr/bin/env python3
"""Run a repeatable VOID inference benchmark and save machine-readable results.

The wrapper is intentionally thin: it uses Netflix VOID's upstream
predict_v2v.py process, records environment metadata, and timestamps the first
"Sequence to run" log line as an approximation of model cold-start load time.
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
from pathlib import Path
from typing import Any


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
    parser.add_argument("--sample-root", type=Path, default=default_repo_root)
    parser.add_argument("--seq-name", default="my_video", help="Sequence folder name to prepare from sample files.")
    parser.add_argument("--run-seqs", default="", help="Comma-separated sequence names. Defaults to --seq-name.")
    parser.add_argument("--data-root", type=Path, default=None, help="VOID data root. Defaults to <void-root>/custom_data.")
    parser.add_argument("--save-path", type=Path, default=default_save_path)
    parser.add_argument("--reports-dir", type=Path, default=default_repo_root / "benchmark" / "reports")
    parser.add_argument("--label", default="void_benchmark")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0, help="Seconds between nvidia-smi samples. Use 0 to disable.")
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


def prepare_sample_data(sample_root: Path, data_root: Path, seq_name: str) -> Path:
    missing = [name for name in REQUIRED_SEQUENCE_FILES if not (sample_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sample files under {sample_root}: {', '.join(missing)}")

    seq_dir = data_root / seq_name
    seq_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_SEQUENCE_FILES:
        shutil.copy2(sample_root / name, seq_dir / name)
    return seq_dir


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
        "max_memory_used_mb": max(sample["memory_used_mb"] for sample in samples),
        "max_utilization_gpu_pct": max(sample["utilization_gpu_pct"] for sample in samples),
    }


def stream_predictor(cmd: list[str], cwd: Path, gpu_sample_interval: float) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    line_tail: list[str] = []
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
        print(line, end="", flush=True)
        line_tail.append(line.rstrip())
        if len(line_tail) > 200:
            line_tail.pop(0)
        if first_sequence_at is None and "Sequence to run:" in line:
            first_sequence_at = time.perf_counter()
    return_code = proc.wait()
    finished = time.perf_counter()

    stop.set()
    if sampler is not None:
        sampler.join(timeout=2)

    return {
        "returncode": return_code,
        "wall_time_s": round(finished - started, 3),
        "time_to_first_sequence_log_s": round(first_sequence_at - started, 3) if first_sequence_at else None,
        "gpu_samples": summarize_gpu_samples(gpu_samples),
        "log_tail": line_tail,
    }


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "void_benchmark"


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    void_root = args.void_root.resolve()
    data_root = (args.data_root or (void_root / "custom_data")).resolve()
    save_path = args.save_path.resolve()
    run_seqs = args.run_seqs or args.seq_name

    if args.setup_void:
        setup_void(void_root)
    if args.download_models:
        download_models(void_root)
    if args.prepare_sample_data:
        prepared = prepare_sample_data(args.sample_root.resolve(), data_root, args.seq_name)
        print(f"Prepared sequence data: {prepared}")

    if not (void_root / "inference" / "cogvideox_fun" / "predict_v2v.py").exists():
        raise FileNotFoundError(f"VOID predictor not found under {void_root}. Use --setup-void first.")
    if not (void_root / BASE_MODEL_DIR / "transformer" / "config.json").exists():
        raise FileNotFoundError(f"Base model is missing under {void_root / BASE_MODEL_DIR}. Use --download-models first.")
    if not (void_root / VOID_CHECKPOINT).exists():
        raise FileNotFoundError(f"VOID checkpoint is missing under {void_root}. Use --download-models first.")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "repo": {
            "root": str(repo_root),
            "branch": git_value(repo_root, "branch", "--show-current"),
            "commit": git_value(repo_root, "rev-parse", "HEAD"),
        },
        "void_root": str(void_root),
        "environment": collect_environment(),
        "benchmark": {
            "run_seqs": run_seqs,
            "data_root": str(data_root),
            "save_path": str(save_path),
            "repeats": args.repeats,
            "extra_config": args.extra_config,
        },
        "runs": [],
    }

    for index in range(args.repeats):
        run_save_path = save_path / f"{sanitize_label(args.label)}_repeat_{index + 1:02d}"
        run_save_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "inference/cogvideox_fun/predict_v2v.py",
            "--config",
            "config/quadmask_cogvideox.py",
            f"--config.data.data_rootdir={data_root}",
            f"--config.experiment.run_seqs={run_seqs}",
            f"--config.experiment.save_path={run_save_path}",
            f"--config.video_model.transformer_path={void_root / VOID_CHECKPOINT}",
            *args.extra_config,
        ]
        result = stream_predictor(cmd, cwd=void_root, gpu_sample_interval=args.gpu_sample_interval)
        result["repeat_index"] = index + 1
        result["save_path"] = str(run_save_path)
        report["runs"].append(result)
        if result["returncode"] != 0:
            break

    gpu_name = "unknown_gpu"
    torch_env = report["environment"].get("torch")
    if isinstance(torch_env, dict):
        gpu_name = torch_env.get("device_name") or gpu_name
    filename = (
        dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        + "_"
        + sanitize_label(args.label)
        + "_"
        + sanitize_label(gpu_name)
        + ".json"
    )
    report_path = args.reports_dir / filename
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBenchmark report written to: {report_path}")

    failed = [run for run in report["runs"] if run["returncode"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
