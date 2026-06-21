# Colab A100 Cold Single-Sequence Benchmark

Date: 2026-06-21

GPU: NVIDIA A100-SXM4-40GB

Scenario: `cold_single_seq`

Sequence: `my_video`

## Controls

- Repeats: 3 per memory mode
- `num_inference_steps`: 30
- `sample_size`: `384x672`
- `max_video_length`: 197
- `temporal_window_size`: 85
- Sequence directory cleaned before each run
- `skip_if_exists=False`

## Median Results

| GPU memory mode | Total wall time | Process to first sequence | Init before transformer log | After first sequence log | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_cpu_offload_and_qfloat8` | 153.5s | 75.1s | 65.2s | 78.5s | 18.9 GiB |
| `model_full_load` | 134.4s | 74.6s | 59.0s | 59.6s | 38.0 GiB |
| `model_cpu_offload` | 147.9s | 68.9s | 59.0s | 79.0s | 18.9 GiB |

## Interpretation

`model_full_load` improves the end-to-end runtime on A100. The median run dropped from 153.5s with the upstream default mode to 134.4s with full model load.

That improvement is mostly after the first sequence starts. It does not materially reduce the cold-start delay before inference begins: the first sequence log is still around 75s with `model_full_load`.

For issue #1, this means `model_full_load` is a useful high-VRAM runtime option, but it is not the startup fix by itself. The next step is finer profiling inside `load_pipeline()`, especially around base CogVideoX transformer loading, VOID checkpoint loading, VAE/T5 loading, and memory-mode setup.

This result is A100-only. A direct L40S run is still required before making L40S-specific claims.

## Source Reports

- `20260621_100350_colab_a100_baseline_cold_single_seq_model_cpu_offload_and_qfloat8_NVIDIA_A100-SXM4-40GB.json`
- `20260621_101134_colab_a100_full_load_cold_single_seq_model_full_load_NVIDIA_A100-SXM4-40GB.json`
- `20260621_101819_colab_a100_cpu_offload_cold_single_seq_model_cpu_offload_NVIDIA_A100-SXM4-40GB.json`
- `20260621_102544_void_benchmark_summary.csv`
