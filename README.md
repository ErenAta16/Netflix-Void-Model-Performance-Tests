# Netflix VOID (`netflix/void-model`) Performance Tests

This repository benchmarks and validates the [Netflix VOID](https://github.com/Netflix/void-model) model (`netflix/void-model`) for object and interaction removal in videos.

The core goal is to evaluate physical causal reasoning in video inpainting. Instead of removing only the visible object, VOID can also update physically related evidence such as shadows, reflections, contact regions, and scene continuity. This is important for realistic scene reconstruction, simulation workflows, and spatial analysis tasks where visual consistency depends on causal relationships.

## What Is Being Tested

All experiments in this repository directly test inference quality of `netflix/void-model` under different physical interaction scenarios:

- large obstacle removal in a dynamic pedestrian scene
- falling-object/contact behavior consistency
- water-surface continuity after object deletion

## Experiment Results

GitHub README pages do not reliably render HTML `<video>` tags.  
For this reason, results are shown as looping GIF previews below. Click any preview to open the original MP4.

### 1) Ice Cream Van Removal (Crowd Scene Obstacle)

This experiment removes a large van blocking a pedestrian path. The model reconstructs plausible pavement structure and preserves crowd motion continuity while removing the object and its interaction footprint.

[![Ice Cream Van Result](my_video-fg=-1-0001_tuple.gif)](my_video-fg=-1-0001_tuple.mp4)

### 2) Lime Physics (Falling Object Interaction)

This experiment tests temporal causality and contact behavior. After object removal, the model maintains believable motion progression and updates object-surface interaction cues.

[![Lime Physics Result](lime-fg=-1-0001_tuple.gif)](lime-fg=-1-0001_tuple.mp4)

### 3) Ducky Float Removal (Water Surface Dynamics)

This experiment focuses on fluid-like surface behavior. The output shows continuity in ripples and reflections as if the removed object had not influenced the water surface.

[![Ducky Float Result](ducky-float-fg=-1-0001_tuple.gif)](ducky-float-fg=-1-0001_tuple.mp4)

## Repository Files

This project is organized into two practical categories.

### A) Output Videos (Model Results)

- `ducky-float-fg=-1-0001_tuple.mp4`
- `lime-fg=-1-0001_tuple.mp4`
- `my_video-fg=-1-0001_tuple.mp4`
- `ducky-float-fg=-1-0001_tuple.gif`
- `lime-fg=-1-0001_tuple.gif`
- `my_video-fg=-1-0001_tuple.gif`

These are inference outputs demonstrating the model's performance on different physical reasoning scenarios.

### B) Source and Setup Files

- `input_video.mp4`: Original source video (ice cream van scene).
- `quadmask_0.mp4`: Interaction-aware 4-value quadmask for the target object.
- `prompt.json`: Background description prompt used during generation.
- `VOID_Inference_Colab.ipynb`: End-to-end Colab notebook for environment setup and inference.
- `benchmark/`: Repeatable GPU benchmark wrapper and protocol for L40S and Colab runtime comparisons.

## How the Source Files Work Together

For each sequence, VOID expects three synchronized inputs:

1. `input_video.mp4` for the original temporal content.
2. `quadmask_0.mp4` for spatial and interaction guidance.
3. `prompt.json` for semantic background reconstruction guidance.

The quadmask encoding used in this project:

- `0`: remove region
- `63`: overlap/boundary region
- `127`: affected interaction region
- `255`: preserve region

This representation helps the model reason about not only where to erase an object, but also which neighboring pixels should be causally updated.

## Step-by-Step Usage

1. Open `VOID_Inference_Colab.ipynb` in Google Colab.
2. Run setup cells to install dependencies and download model checkpoints.
3. Upload `input_video.mp4`, `quadmask_0.mp4`, and `prompt.json` when prompted.
4. Run the inference cell (Pass 1) to generate outputs.
5. Review generated videos under the configured output directory.

## L40S Cold-Start Note

Issue [#1](https://github.com/ErenAta16/Netflix-Void-Model-Performance-Tests/issues/1) tracks a startup bottleneck where L40S runs spend about 40-50 seconds loading the CogVideoX 5B transformer before inference begins.

The Colab notebook now treats that as cold-start overhead and separates it from the rest of the predictor wall time:

- Re-running setup reuses the `/content/void-model` checkout instead of deleting and cloning it every time.
- `hf_transfer` is enabled for faster Hugging Face checkpoint downloads.
- The upload cell builds `RUN_SEQS` from every valid folder under `/content/void-model/custom_data`.
- The inference cell supports `cold_single_seq` for per-job latency and `batched_multi_seq` for amortized throughput.
- The inference cell exposes `GPU_MEMORY_MODE`, so high-VRAM GPUs can compare `model_full_load` against the upstream `model_cpu_offload_and_qfloat8` default.
- The inference cell streams predictor logs and prints total wall time, initialization before the transformer-load log, the transformer-load-to-sequence bracket, and the remaining wall time after the first sequence marker.

A Colab `my_video` run reported `VOID predictor wall time: 175.7s`. That number is total predictor wall time for the single sequence, not cold-start time by itself. Use the updated notebook or `benchmark/run_void_benchmark.py` to split that total into cold-start/loading time and post-load inference time.

The current A100 result shows the larger delay happens before the `Load transformer from checkpoint` log, so issue #1 is best treated as a predictor initialization and memory-mode problem rather than pure safetensors I/O. A100 results are diagnostic only; L40S results should be measured directly before claiming an L40S fix.

This does not remove the initial load cost, but it prevents paying the same cost once per sequence when benchmarking multiple VOID cases. For a single sequence, the next fix candidate is `model_full_load` on high-VRAM GPUs. For multiple sequences, prepare all sequences first and run one predictor process.

For repeatable timing, the notebook recreates the uploaded sequence directory and writes each run to a timestamped output directory. This prevents stale masks from adding extra foreground runs and prevents old outputs from being skipped.

For structured GPU comparisons, use `benchmark/run_void_benchmark.py`. It records the actual GPU name, total predictor wall time, and time to the first inference sequence log in a JSON report. See `benchmark/README.md` for the L40S and Colab test matrix.

## Runtime Requirements

- Google Colab or equivalent Linux environment
- Python dependencies installed by the notebook (`huggingface_hub`, model requirements, and system packages such as `ffmpeg`)

## Credits

- Model under test: [Netflix VOID (`netflix/void-model`)](https://github.com/Netflix/void-model)
- Base diffusion/video stack used by VOID: CogVideoX-related components as defined in the original repository

