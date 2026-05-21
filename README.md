# TelePhysics

**[Project Page](https://telephysics.github.io/) | [Paper](assets/paper.pdf) | [arXiv](https://arxiv.org/abs/2605.20290)** 

TelePhysics is a unified, training-free framework designed to facilitate holistic 3D scene generation and
physically grounded video synthesis from a single input image. The figure showcases interactions among multiple
objects across diverse scene

Supported physics materials: rigid bodies, elastic solids (MPM), sand (MPM), elastoplastic (MPM), cloth (PBD), and liquid (SPH).

---

## Pipeline Overview

![Pipeline Overview](assets/pipeline.jpg)

---

## Requirements

- Linux (tested on Ubuntu 20.04+)
- CUDA 12.6
- Conda

---

## Installation

```bash
bash environments/install.sh
```

This creates two conda environments:

| Environment | Purpose |
|---|---|
| `telephysics-pq` | Segmentation + 3D mesh generation (PyTorch 2.8.0 + CUDA 12.6) |
| `telephysics-sr` | Physics simulation + depth estimation + video synthesis |

---

## Model Download

```bash
bash environments/download.sh
```

Downloads the following models into `./models/`:

| Model | Purpose |
|---|---|
| `facebook/sam3` | Text-prompted image segmentation |
| `facebook/sam-3d-objects` | Single-image 3D mesh reconstruction |
| `facebookresearch/dinov2` | Vision backbone for SAM3D |
| `Ruicheng/moge-vitl` | Monocular geometry estimation |
| `depth-anything/Video-Depth-Anything` | Per-frame depth estimation (ViT-S / ViT-L) |
| `smartywu/big-lama` | Background inpainting |
| `PAI/Wan2.2-VACE-Fun-A14B` | Video synthesis (high/low noise denoiser) |
| `DiffSynth-Studio/Wan-Series-Converted-Safetensors` | VAE + T5 encoder |
| `Wan-AI/Wan2.1-T2V-1.3B` | UMT5-XXL text tokenizer |

---

## Quick Start

Run the full pipeline on the provided `ball` example:

```bash
bash scripts/run.sh
```

Output video: `demo/output_ball/wan/rendered_ball.mp4`

To test with your own image, edit the variables at the top of `scripts/run.sh`:

```bash
ROOT_DIR="data"          # directory containing scene folder
NAME="ball"              # scene name (folder name and image stem)
TEXT_PROMPT="ball"       # space-separated object text prompts
OUTPUT_DIR="demo/output_${NAME}"
MOVE=0                   # camera movement: 0=static, 1-4=orbit, 5=dolly-out, 6=dolly-in
```

Place your image at `data/{NAME}/{NAME}.png`. The pipeline will auto-generate a `config.yaml` for the scene.

---

## Data Layout

```
data/
└── {scene_name}/
    ├── {scene_name}.png     # input image
    └── config.yaml          # physics configuration (auto-generated or manual)
```

Three example scenes are included: `ball`, `dress`, and `sandhouse`.

---

## Scene Configuration

Each scene is controlled by `data/{scene_name}/config.yaml`. All fields are optional; defaults are applied automatically.

```yaml
simulation:
  n_steps: 300          # total simulation steps
  fps: 60               # output frame rate
  camera_mv: 0          # camera movement (0=static, 1-6=various motions)

objects:
  0:
    material: "rigid"   # rigid | mpm_elastic | mpm_elastoplastic |
                        # mpm_sand | pbd_cloth | sph_liquid
    fixed: false
    start_frame: 0
    velocity: [0.0, 0.0, 0.0]   # initial linear velocity (m/s)

forces:
  - type: "wind"
    direction: [1, 0, 0]
    strength: 5.0
```

See [`configs/example_config.yaml`](configs/example_config.yaml) for the full reference with all available options, including material parameters, force fields (constant, wind, point, drag, noise, vortex, turbulence), and camera alignment controls.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{zhang2026telephysicsphysicsgroundedmultiobjectscene,
      title={TelePhysics: Physics-Grounded Multi-Object Scene Generation from a Single Image with Real-Time Interaction}, 
      author={Xin Zhang and Yabo Chen and Yijie Fang and Wanying Qu and Haibin Huang and Chi Zhang and Feng Xu and Xuelong Li},
      year={2026},
      eprint={2605.20290},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2605.20290}, 
}
```

---

## License

See individual model licenses for third-party components. The TelePhysics pipeline code is released under the [MIT License](LICENSE).

---

## Acknowledgements

We thank the following open-source projects that made this work possible:

- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) — Monocular depth estimation
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) — Single-image 3D object reconstruction
- [SAM3](https://github.com/facebookresearch/sam3) — Text-prompted image segmentation
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — Video synthesis pipeline (Wan2.2-VACE)
- [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) — Per-frame video depth estimation
