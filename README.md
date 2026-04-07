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

## Runtime Requirements

- Google Colab or equivalent Linux environment
- Python dependencies installed by the notebook (`huggingface_hub`, model requirements, and system packages such as `ffmpeg`)

## Credits

- Model under test: [Netflix VOID (`netflix/void-model`)](https://github.com/Netflix/void-model)
- Base diffusion/video stack used by VOID: CogVideoX-related components as defined in the original repository

