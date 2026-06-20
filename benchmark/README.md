# VOID GPU Benchmark Protocol

This benchmark is for issue #1: measuring how much time VOID spends before inference starts, especially while loading the CogVideoX 5B transformer on L40S-class GPUs.

## What We Measure

`benchmark/run_void_benchmark.py` runs Netflix VOID's upstream `inference/cogvideox_fun/predict_v2v.py` and saves a JSON report with:

- GPU name and VRAM from PyTorch and `nvidia-smi`
- driver, CUDA, Python, and Torch metadata
- total predictor wall time
- time to the first `Sequence to run:` log line, which approximates cold-start model load time
- optional sampled peak GPU memory and utilization
- command log tail for debugging

## Recommended Test Matrix

Run the same sequence and command on every GPU so the results are comparable.

1. L40S 48 GB: primary issue target.
2. Colab A100, if assigned.
3. Colab L4, if assigned.
4. Colab T4, if assigned, expected to be memory constrained or slower.

Normal Colab does not guarantee a specific GPU type. Record the actual GPU from the benchmark JSON instead of assuming the runtime type.

## L40S Run

Use a provider that offers a real NVIDIA L40S 48 GB instance, then run:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git
git clone https://github.com/ErenAta16/Netflix-Void-Model-Performance-Tests.git
cd Netflix-Void-Model-Performance-Tests
python benchmark/run_void_benchmark.py \
  --setup-void \
  --download-models \
  --prepare-sample-data \
  --label l40s_my_video \
  --repeats 1
```

For repeat measurements after the first setup, omit `--setup-void --download-models`:

```bash
python benchmark/run_void_benchmark.py \
  --prepare-sample-data \
  --label l40s_my_video_repeat \
  --repeats 3
```

## Colab Runs

Open `VOID_Inference_Colab.ipynb`, choose a GPU runtime, and run setup/download/upload first. Then clone this repo in a final code cell and run:

```python
!git clone https://github.com/ErenAta16/Netflix-Void-Model-Performance-Tests.git /content/Netflix-Void-Model-Performance-Tests
%cd /content/Netflix-Void-Model-Performance-Tests
!python benchmark/run_void_benchmark.py \
  --void-root /content/void-model \
  --prepare-sample-data \
  --sample-root /content/Netflix-Void-Model-Performance-Tests \
  --label colab_my_video \
  --repeats 1
```

If Colab assigns a different GPU after reconnecting, keep the JSON report from each run. The GPU name in the report is the source of truth.

## Commit Policy

Use measurement reports to decide follow-up commits:

- If L40S `time_to_first_sequence_log_s` is near 40-50 seconds but multi-sequence wall time improves, keep the batch/amortization workflow.
- If Colab GPUs fail due to memory, document the failing GPU and adjust notebook defaults rather than hiding the failure.
- If A100/L40S pass but T4/L4 fail, keep the runtime requirement explicit and add a compatibility table.
- Commit benchmark JSON summaries only when they are small enough to review. Do not commit generated videos from benchmark reruns unless the visual output changed intentionally.
