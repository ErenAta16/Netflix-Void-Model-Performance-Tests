# VOID GPU Benchmark Protocol

This benchmark is for issue #1: measuring how much time VOID spends before inference starts, especially while loading the CogVideoX 5B transformer on L40S-class GPUs.

## What We Measure

`benchmark/run_void_benchmark.py` runs Netflix VOID's upstream `inference/cogvideox_fun/predict_v2v.py` and saves a JSON report with:

- GPU name and VRAM from PyTorch and `nvidia-smi`
- driver, CUDA, Python, and Torch metadata
- total predictor wall time
- `init_before_transformer_s`: first predictor output to the `Load transformer from checkpoint` log
- `load_transformer_to_sequence_s`: the bracket between `Load transformer from checkpoint` and `Sequence to run`
- `time_to_first_sequence_log_s`: process start to first sequence log
- `post_first_sequence_log_s`: remaining wall time after the first sequence log
- optional sampled peak GPU memory and utilization
- command line, commit hashes, config overrides, and log tail for debugging

The `load_transformer_to_sequence_s` value is not pure safetensors I/O. It is only the time between two observable log lines. In the A100 Colab run, that bracket was small; the larger cost was earlier initialization and memory-mode setup.

The benchmark recreates the prepared sample sequence by default and forces `--config.experiment.skip_if_exists=False`. This avoids two common measurement errors:

- stale `quadmask_*.mp4` files can make VOID run extra foreground IDs for the same sequence
- old outputs can make VOID skip work and underreport runtime

## Recommended Test Matrix

Keep latency and throughput separate:

1. `cold_single_seq`: one sequence in a fresh predictor process. This matches issue #1's per-job startup complaint.
2. `batched_multi_seq`: every prepared sequence in one predictor process. This measures amortized throughput and should not be used as single-job latency.

For each scenario, test the memory modes below on the same GPU:

- `model_cpu_offload_and_qfloat8`: upstream default and baseline
- `model_full_load`: first candidate for A100 40 GB and L40S 48 GB
- `model_cpu_offload`: fallback if full load OOMs
- `sequential_cpu_offload`: low-memory fallback only

Keep these controls fixed while comparing memory modes:

- `--num-inference-steps 30`
- `--sample-size 384x672`
- `--max-video-length 197`
- `--temporal-window-size 85`

Run each configuration at least three times and compare median plus min/max. Colab and OS page cache can make single-run results noisy. A100 measurements are useful for diagnosis, but they are not a substitute for L40S measurements because A100 and L40S differ in FP8 support.

## Cold Single-Sequence Run

Use this first. It answers the issue directly.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git
git clone https://github.com/ErenAta16/Netflix-Void-Model-Performance-Tests.git
cd Netflix-Void-Model-Performance-Tests
python benchmark/run_void_benchmark.py \
  --setup-void \
  --download-models \
  --prepare-sample-data \
  --scenario cold_single_seq \
  --gpu-memory-mode model_cpu_offload_and_qfloat8 \
  --label baseline_cold_single \
  --repeats 3
```

Then test the high-VRAM candidate:

```bash
python benchmark/run_void_benchmark.py \
  --prepare-sample-data \
  --scenario cold_single_seq \
  --gpu-memory-mode model_full_load \
  --label full_load_cold_single \
  --repeats 3
```

If `model_full_load` OOMs, retry with:

```bash
python benchmark/run_void_benchmark.py \
  --prepare-sample-data \
  --scenario cold_single_seq \
  --gpu-memory-mode model_cpu_offload \
  --label cpu_offload_cold_single \
  --repeats 3
```

Each repeat is written to the JSON report immediately after it finishes. A later OOM does not discard earlier successful measurements.

If you intentionally want to keep extra files in the target sequence directory, pass `--no-clean-seq-dir`. Do not use that flag for issue #1 latency measurements.

## Batched Throughput Run

Use this only after every sequence folder is prepared under `custom_data/`.

```bash
python benchmark/run_void_benchmark.py \
  --scenario batched_multi_seq \
  --gpu-memory-mode model_full_load \
  --label full_load_batched \
  --repeats 3
```

This measures amortized throughput. It will hide per-job cold-start latency by spreading the startup cost over multiple sequences.

## Colab Runs

Open `VOID_Inference_Colab.ipynb`, choose a GPU runtime, and run setup/download/upload first. The notebook exposes:

- `BENCHMARK_SCENARIO`
- `GPU_MEMORY_MODE`
- `NUM_INFERENCE_STEPS`
- `SAMPLE_SIZE`
- `MAX_VIDEO_LENGTH`
- `TEMPORAL_WINDOW_SIZE`

For structured Colab runs, clone this repo in a final code cell and run:

```python
!git clone https://github.com/ErenAta16/Netflix-Void-Model-Performance-Tests.git /content/Netflix-Void-Model-Performance-Tests
%cd /content/Netflix-Void-Model-Performance-Tests
!python benchmark/run_void_benchmark.py \
  --void-root /content/void-model \
  --prepare-sample-data \
  --sample-root /content/Netflix-Void-Model-Performance-Tests \
  --scenario cold_single_seq \
  --gpu-memory-mode model_full_load \
  --label colab_a100_full_load_cold_single \
  --repeats 3
```

If Colab assigns a different GPU after reconnecting, keep the JSON report from each run. The GPU name in the report is the source of truth.

### Interpreting the Current Colab Result

The current `my_video` Colab run printed:

```text
Running sequences in one predictor process: my_video
The first part of this wall time includes CogVideoX transformer cold-start loading.
VOID predictor wall time: 175.7s
```

Because that notebook cell used `subprocess.run`, the `175.7s` number is total predictor wall time. It includes cold-start loading, denoising, decoding, and output writing. It should not be reported as the transformer load time by itself.

The updated notebook and benchmark wrapper stream predictor output and timestamp the first `Sequence to run:` line. Use that marker to split the total into:

- `time_to_first_sequence_log_s`: approximate model setup and cold-start time.
- `post_first_sequence_log_s`: remaining predictor time after sequence processing begins.

For one sequence, the cold-start cost is unavoidable. For multiple videos, prepare every sequence first and run them in one predictor process so the CogVideoX transformer is loaded once and reused across all prepared sequences.

The A100 `model_full_load` smoke run did not OOM, but it reused a stale Colab sequence/output directory and ran extra foreground IDs. Treat that run as a methodology check only. Re-run after the notebook update so the sequence directory and output path are clean.

## Summaries

Create CSV and Markdown summaries from JSON reports:

```bash
python benchmark/summarize_reports.py
```

The summary table is intended for issue comments and pull request descriptions.

## Commit Policy

Use measurement reports to decide follow-up commits:

- If `init_before_transformer_s` dominates, the issue is broad predictor initialization and memory-mode setup, not safetensors I/O.
- If `model_full_load` reduces `init_before_transformer_s` and total wall time without OOM on A100/L40S, document it as the recommended high-VRAM mode.
- If A100 improves but L40S data is missing, do not extrapolate. Mark L40S results as pending.
- If Colab GPUs fail due to memory, document the failing GPU and adjust notebook defaults rather than hiding the failure.
- If A100/L40S pass but T4/L4 fail, keep the runtime requirement explicit and add a compatibility table.
- Commit benchmark JSON summaries only when they are small enough to review. Do not commit generated videos from benchmark reruns unless the visual output changed intentionally.
