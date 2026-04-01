#!/usr/bin/env bash
set -e

# ============================================================
# TelePhysics — Model Download Script
# ============================================================

pip install -U huggingface_hub

MODELS_DIR="./models"
mkdir -p "${MODELS_DIR}"

HF_CMD="huggingface-cli download --local-dir-use-symlinks False"

################ SAM3 ################
# Image segmentation model
${HF_CMD} facebook/sam3 \
  --local-dir "${MODELS_DIR}/SAM3"

################ SAM3D (hf) ################
# 3D object mesh generation
${HF_CMD} facebook/sam-3d-objects \
  --local-dir "${MODELS_DIR}/SAM3D/hf"

################ SAM3D (dinov2) ################
# Vision backbone for SAM3D
DINOV2_DIR="${MODELS_DIR}/SAM3D/facebookresearch_dinov2_main"
if [ ! -d "${DINOV2_DIR}" ]; then
  git clone https://github.com/facebookresearch/dinov2.git "${DINOV2_DIR}"
fi
wget -nc -O "${DINOV2_DIR}/dinov2_vitl14_reg4_pretrain.pth" \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth

################ VDA (Video-Depth-Anything) ################
# Monocular video depth estimation
mkdir -p "${MODELS_DIR}/VDA"
wget -nc -P "${MODELS_DIR}/VDA" \
  https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
wget -nc -P "${MODELS_DIR}/VDA" \
  https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth

################ DiffSynth-Studio ################
# VAE & T5 encoder for video synthesis
${HF_CMD} DiffSynth-Studio/Wan-Series-Converted-Safetensors \
  --local-dir "${MODELS_DIR}/DiffSynth-Studio/Wan-Series-Converted-Safetensors"

################ PAI ################
# Wan2.2-VACE video generation (high/low noise denoiser)
${HF_CMD} PAI/Wan2.2-VACE-Fun-A14B \
  --local-dir "${MODELS_DIR}/PAI/Wan2.2-VACE-Fun-A14B"

################ Wan-AI ################
# Tokenizer for video synthesis pipeline
${HF_CMD} Wan-AI/Wan2.1-T2V-1.3B \
  --include "google/umt5-xxl/*" \
  --local-dir "${MODELS_DIR}/Wan-AI/Wan2.1-T2V-1.3B"

################ MoGe ################
# Monocular geometry estimation (HF cache format for sam3d_objects)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Ruicheng/moge-vitl', cache_dir='${MODELS_DIR}')
"

################ LAMA (big-lama) ################
# Inpainting model for background removal
if [ ! -d "${MODELS_DIR}/big-lama" ]; then
  curl -L -o "${MODELS_DIR}/big-lama.zip" \
    https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
  unzip -o "${MODELS_DIR}/big-lama.zip" -d "${MODELS_DIR}"
  rm -f "${MODELS_DIR}/big-lama.zip"
fi

echo "All models downloaded successfully."

