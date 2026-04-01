#!/usr/bin/env bash
set -e

# ============================================================
# TelePhysics — Environment Setup
# ============================================================
# Creates two conda environments:
#   1. telephysics-pa    — Segmentation, 3D mesh generation (PyTorch 2.8.0 + CUDA 12.6)
#   2. telephysics-sr — Physics simulation, depth estimation, video synthesis
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ----------------------------------------------------------
# Environment 1: telephysics (Segmentation + Mesh Generation)
# ----------------------------------------------------------
echo ">>> Creating conda environment: telephysics"

conda env create -f "${PROJECT_DIR}/environments/default.yml"
conda activate telephysics-pa

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu126.html"
pip install -r "${PROJECT_DIR}/environments/requirements-pq.txt"

pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git"
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

# mip-splatting (diff-gaussian-rasterization)
TMPDIR=$(mktemp -d)
git clone --recursive https://github.com/autonomousvision/mip-splatting.git "${TMPDIR}/mip-splatting"
cd "${TMPDIR}/mip-splatting/submodules/diff-gaussian-rasterization"
pip install . --no-build-isolation
cd "${PROJECT_DIR}"
rm -rf "${TMPDIR}"

echo ">>> telephysics environment ready."

# ----------------------------------------------------------
# Environment 2: telephysics-sr (Simulation + Rendering)
# ----------------------------------------------------------
echo ">>> Creating conda environment: telephysics-sr"

conda create -y -n telephysics-sr python=3.12
conda activate telephysics-sr

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install genesis-world==0.3.14
pip install -r "${PROJECT_DIR}/environments/requirements-sr.txt"

echo ">>> telephysics-sr environment ready."
echo ">>> All environments installed successfully."
