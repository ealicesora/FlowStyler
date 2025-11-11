# FlowStyler Video Style Transfer

Lagrangian particle-based video style transfer pipeline rewritten as a modular Python project. The pipeline composes stylized frames with dense optical flow guidance and temporal regularization.

## Highlights

- Trajectory-aware field system that keeps textures temporally consistent.
- RAFT-based bidirectional optical flow with occlusion masks.
- Optional per-scene color matching and temporal post-processing.
- Configurable training schedule, regularization, and data preprocessing.

## Publication

- Paper: [FlowStyler: Artistic Video Stylization via Transformation Fields Transports](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_FlowStyler_Artistic_Video_Stylization_via_Transformation_Fields_Transports_ICCV_2025_paper.pdf)
- If you use FlowStyler in academic work, please cite:

```bibtex
@InProceedings{Gong_2025_ICCV,
    author    = {Gong, Yuning and Chen, Jiaming and Ren, Xiaohua and Liao, Yuanjun and Zhang, Yanci},
    title     = {FlowStyler: Artistic Video Stylization via Transformation Fields Transports},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {10229-10238}
}
```

## Requirements

- Python 3.10+ (PyTorch 2.9.0 CUDA build recommended)
- CUDA-capable GPU with ≥8 GB memory
- System packages for OpenCV and ffmpeg (ffmpeg is required for video export)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Pretrained Weights

Download the following checkpoints and place them under `models/` (the filenames must stay unchanged):

| Component | Filename | Source |
| --- | --- | --- |
| RAFT optical flow | `models/raft_large_C_T_SKHT_V2-ff5fadd5.pth` | [RAFT (ECCV 2020)](https://github.com/princeton-vl/RAFT) community weights |
| VGG feature extractor | `models/vgg16-397923af.pth` | [PyTorch model zoo](https://download.pytorch.org/models/vgg16-397923af.pth) |

> Tip: keep the weights under version control with Git LFS or provide download scripts for contributors.

## Data Preparation

The training script expects a folder containing per-frame RGB images:

```
<data_dir>/<video_name>/
    frame0001.jpg
    frame0002.jpg
    ...
```

By default, the repository points to DAVIS2017 frames. Update paths in `config/default.yaml` to match your dataset.

## Configuration

All options are managed through YAML configs (see `config/default.yaml`):

- `video`: input/output paths, optional resizing, FPS of exported video.
- `style`: style image, strength weights, and preprocessing hints.
- `optical_flow`: RAFT parameters and mask erosion.
- `optimization`: learning rates, step counts, flip mode, and optimizer choice.
- `regularization`: smoothness penalties for warp fields.
- `color_matching`: per-scene color transfer toggle.
- `strategy`: occlusion-aware updates and temporal blending options.
- `postprocess`: temporal brightness smoothing applied before saving results.

Clone and adjust the default config for each experiment:

```bash
cp config/default.yaml config/my_scene.yaml
vim config/my_scene.yaml
```

## Training

Launch the end-to-end training loop:

```bash
python scripts/train.py --config config/my_scene.yaml
```

Key outputs:

- Stylized frames saved to `results/<video_name>_<style_image>/`.
- A temporal video summary at `results/<video_name>_<style_image>/output_resized.mp4`.

The script prints progress for every epoch/frame and reports the final output directory.


## Repository Layout

```
config/          YAML experiment configs
images/styles/   Example style images
lib/             Low-level field, flow, and optimization helpers
models/          Pretrained weights (download separately)
scripts/         Training and inference entry points
src/             Modular data, model, optimization, and utility code
utils/           Additional image and math utilities
```

## Troubleshooting

- **CUDA OOM**: reduce resolution via `video.enable_resize` or lower `optimization.num_steps`.
- **Missing weights**: verify both `.pth` files exist under `models/` before training starts.
- **Inconsistent colors**: disable `color_matching.enable` or tune `alpha` / `beta`.
- **Jitter artifacts**: increase `regularization.lambda_jacobian` or enable `postprocess.smooth_brightness`.

