# TelePhysics

**[Project Page](https://telephysics.github.io/) | [Paper](assets/paper.pdf)**

TelePhysics is an image-to-physics-video pipeline. Given a single photograph and a text prompt, it automatically:

1. Segments and inpaints objects (SAM3 + LaMa)
2. Reconstructs 3D meshes (SAM3D + MoGe)
3. Runs physically-based simulation (Genesis)
4. Estimates per-frame depth (Video-Depth-Anything)
5. Synthesizes a photorealistic video (Wan2.2-VACE)

Supported physics materials: rigid bodies, elastic solids (MPM), sand (MPM), elastoplastic (MPM), cloth (PBD), and liquid (SPH).

---

## Pipeline Overview

![Pipeline Overview](assets/pipeline.jpg)

```
Input Image
    │
    ▼
[Step 1] SAM3 Segmentation + LaMa Inpainting
    │
    ▼
[Step 2] SAM3D 3D Mesh Generation
    │
    ▼
[Step 3] Genesis Physics Simulation  ──► simulation frames (PNG)
    │
    ▼
[Step 4] Video-Depth-Anything Depth Estimation
    │
    ▼
[Step 5] Wan2.2-VACE Video Synthesis  ──► rendered video (MP4)
```

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
@article{telephysics,
  title   = {TelePhysics},
  year    = {2025},
}
```

---

## License

See individual model licenses for third-party components. The TelePhysics pipeline code is released under the [MIT License](LICENSE).
