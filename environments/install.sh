#!/usr/bin/env bash
set -e

# ============================================================
# PhysOmni — Environment Setup
# ============================================================
# Creates two conda environments:
#   1. physomni-pq — Segmentation, 3D mesh generation (PyTorch 2.8.0 + CUDA 12.6)
#   2. physomni-sr — Physics simulation, depth estimation, video synthesis
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ----------------------------------------------------------
# Environment 1: physomni-pq (Segmentation + Mesh Generation)
# ----------------------------------------------------------
echo ">>> Creating conda environment: physomni-pq"

conda env create -f "${PROJECT_DIR}/environments/default.yml"
conda activate physomni-pq

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

echo ">>> physomni-pq environment ready."

# ----------------------------------------------------------
# Environment 2: physomni-sr (Simulation + Rendering)
# ----------------------------------------------------------
echo ">>> Creating conda environment: physomni-sr"

conda create -y -n physomni-sr python=3.12
conda activate physomni-sr

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install genesis-world==0.3.14
pip install -r "${PROJECT_DIR}/environments/requirements-sr.txt"

echo ">>> physomni-sr environment ready."
echo ">>> All environments installed successfully."
