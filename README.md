# FlowStyler: Artistic Video Stylization via Transformation Fields Transports

FlowStyler is the official implementation of  
**“FlowStyler: Artistic Video Stylization via Transformation Fields Transports” (ICCV 2025)**.

It is a modular, PyTorch-based, **non-generative** video style transfer framework that achieves
highly artistic stylization with strong temporal consistency and practical GPU memory usage.

> TL;DR  
> Instead of running diffusion over the whole video, FlowStyler optimizes two transport fields
> across frames—**stylization velocity field** and **orthogonality-regularized color transfer field**—
> guided by RAFT optical flow, momentum-preserving updates, and occlusion-aware lookup,
> producing flicker-free, expressive video stylization.

Online demo: https://www.bilibili.com/video/BV1dt3czGEeh
---

## Key Features

- **Field-based stylization**
  - Reformulates video stylization as the joint evolution of:
    - A **stylization velocity field** for geometric deformation.
    - An **orthogonality-regularized color transfer field** for stable, expressive color changes.
  - Moves beyond naive per-frame optimization and simple flow warping.

- **Lagrangian, trajectory-aware transport**
  - Treats pixels as particles advected by the velocity field over time.
  - Reduces distortion and misalignment issues common in direct pixel or flow-based warping.

- **Momentum-preserving optimization**
  - Propagates optimizer states across frames.
  - Suppresses vibration artifacts while preserving high-frequency style details.

- **Occlusion-aware temporal lookup**
  - Incorporates multi-frame, mask-normalized lookup.
  - Handles disocclusions and avoids motion trails and ghosting.

- **Practical & lightweight**
  - Frame-wise optimization with RAFT flow.
  - Stable memory behavior suitable for high-end consumer GPUs.

---

## Getting Started

### Requirements

- Python **3.10+**
- PyTorch **2.x** with CUDA
- CUDA-capable GPU with **≥ 8 GB** memory recommended
- `ffmpeg` (for video export)
- OpenCV-related system libraries as required by your environment

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Pretrained Weights

Place the following checkpoints under `models/` (filenames must match):

| Component         | Filename                               | Source                     |
| ----------------- | -------------------------------------- | -------------------------- |
| RAFT optical flow | raft_large_C_T_SKHT_V2-ff5fadd5.pth   | RAFT official/community    |
| VGG-16 features   | vgg16-397923af.pth                    | PyTorch model zoo          |

You may manage these with Git LFS or add a helper download script if desired.

---

## Data Preparation

FlowStyler operates on per-frame RGB images.

Expected directory layout:

```text
<data_root>/<video_name>/
    frame0001.jpg
    frame0002.jpg
    ...
```

You can:
1. Extract frames from your input video.
2. Update the paths in your config file accordingly.

---

## Configuration

All options are managed through YAML configs.  
See `config/default.yaml` as a reference.

Key sections:

- **`video`**
  - Input frame directory, output directory, optional resizing, FPS for exported video.

- **`style`**
  - Style image path, style weights, and optional preprocessing hints.

- **`optical_flow`**
  - RAFT weights path, flow resolution, forward–backward consistency checks, occlusion masking.

- **`optimization`**
  - Iterations per frame, learning rates for different fields, optimizer choice, update schedule.

- **`regularization`**
  - Smoothness and TV-like penalties for the velocity field.

- **`color_matching`**
  - Global or per-scene color transfer toggles and parameters.

- **`strategy`**
  - Momentum-preserving updates, field propagation strategy, occlusion-aware blending.

- **`postprocess`**
  - Temporal brightness smoothing and minor stabilization before saving results.

Create your own experiment config:

```bash
cp config/default.yaml config/my_scene.yaml
vim config/my_scene.yaml
```

---

## Running FlowStyler

Launch the end-to-end stylization pipeline:

```bash
python scripts/train.py --config config/my_scene.yaml
```

Typical outputs:

- Stylized frames:
  ```text
  results/<video_name>_<style_name>/frames/*.jpg
  ```
- Composited video:
  ```text
  results/<video_name>_<style_name>/output_resized.mp4
  ```

The script reports:
- Optimization progress per frame.
- Key loss terms (content/style/regularization).
- Final output directory.

Note: FlowStyler performs **per-video, per-style optimization**, not universal feed-forward inference.

---

## Repository Structure

```text
config/           YAML experiment configs
images/styles/    Example style images
lib/              Low-level field, flow, and optimization utilities
models/           Pretrained weights (to be downloaded by user)
scripts/          Entry points for stylization / experiments
src/              Core implementation: data IO, fields, losses, pipeline
utils/            Logging, visualization, and math helpers
```

---

## Troubleshooting

- **CUDA out-of-memory**
  - Enable resizing in `video` section.
  - Reduce `optimization.num_steps` or use smaller resolutions.

- **Missing weights**
  - Confirm RAFT and VGG checkpoints exist under `models/`.
  - Ensure config paths match the actual filenames.

- **Overly strong or distorted colors**
  - Lower style weight in `style`.
  - Adjust or disable `color_matching`.
  - Tune color field regularization.

- **Temporal flicker / jitter**
  - Verify RAFT weights and occlusion masks are correctly configured.
  - Increase velocity field regularization.
  - Enable or strengthen momentum-preserving and occlusion-aware strategies in `strategy`.
  - Use `postprocess` smoothing for subtle stabilization.

---

## Citation

If you find FlowStyler useful in your research, please cite:

```bibtex
@InProceedings{Gong_2025_ICCV,
    author    = {Gong, Yuning and Chen, Jiaming and Ren, Xiaohua and Liao, Yuanjun and Zhang, Yanci},
    title     = {FlowStyler: Artistic Video Stylization via Transformation Fields Transports},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {10229--10238}
}
```
