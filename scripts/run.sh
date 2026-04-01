#!/bin/bash

# ============================================================
# TelePhysics — Full Pipeline
# ============================================================
# Usage:
#   bash run.sh
#
# Edit the variables below to match your setup.
# ============================================================

set -e
set -o pipefail

echo "========== TelePhysics Pipeline =========="

# --------------------------
# Project Root & PYTHONPATH
# --------------------------
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/third_party"
cd "$PROJECT_DIR"

# --------------------------
# User Configuration
# --------------------------

# Root directory containing the scene folder (must contain $NAME/$NAME.png)
ROOT_DIR="data"

# Scene name — corresponds to folder $ROOT_DIR/$NAME/ and image $NAME.png
NAME="ball"

# Text prompt(s) for SAM3 segmentation (space-separated for multiple objects)
TEXT_PROMPT="ball"

# Output directory for simulation video and depth maps
OUTPUT_DIR="demo/output_${NAME}"

# Camera movement type: 0=static, 1-4=circular orbit, 5=dolly-out, 6=dolly-in
MOVE=0

# --------------------------
# Conda Configuration
# --------------------------

# Path to conda.sh — update if conda is installed elsewhere
CONDA_PATH="$(conda info --base)/etc/profile.d/conda.sh"

# --------------------------
# Activate Conda
# --------------------------
echo ">>> Activating Conda Environment..."

if [ ! -f "$CONDA_PATH" ]; then
    echo "[ERROR] conda.sh not found at: $CONDA_PATH"
    echo "  Set CONDA_PATH to the correct path, e.g.:"
    echo "  CONDA_PATH=\"\$HOME/miniconda3/etc/profile.d/conda.sh\""
    exit 1
fi

source "$CONDA_PATH"
conda activate telephysics-pq

echo "Using Python: $(which python)"
echo "Using CUDA:   $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Torch not found')"

# --------------------------
# Step 1: SAM Segmentation & Inpainting
# # --------------------------
echo ""
echo ">>> Step 1: SAM Segmentation & Inpainting"
python pipeline/segmentation.py \
    --root_dir "$ROOT_DIR" \
    --obj_name "$NAME" \
    --text_prompt $TEXT_PROMPT

echo "[Done] Step 1"

# --------------------------
# Step 2: 3D Mesh Generation
# --------------------------
echo ""
echo ">>> Step 2: 3D Mesh Generation"
python pipeline/mesh_gen.py \
    --root_dir "$ROOT_DIR" \
    --scenes_name "$NAME"

echo "[Done] Step 2"

# --------------------------
# Switch to simulation environment
# --------------------------
conda activate telephysics-sr

echo "Using Python: $(which python)"

# --------------------------
# Step 3: Physics Simulation
# --------------------------
echo ""
echo ">>> Step 3: Physics Simulation"
python pipeline/simulation.py \
    --root_dir "$ROOT_DIR" \
    --scene_name "$NAME" \
    --output_dir "$OUTPUT_DIR" \
    --move "$MOVE"

echo "[Done] Step 3"

# --------------------------
# Step 4: Depth Estimation
# --------------------------
echo ""
echo ">>> Step 4: Depth Estimation"
python pipeline/video_to_depth.py \
    --root_dir "$OUTPUT_DIR" \
    --scene_name "$NAME"

echo "[Done] Step 4"

# --------------------------
# Step 5: Video Synthesis
# --------------------------
echo ""
echo ">>> Step 5: Video Synthesis (Wan2.2-VACE)"
python pipeline/video_synthesis.py \
    --base_dir "$OUTPUT_DIR" \
    --scene_name "$NAME" \
    --root_dir "$ROOT_DIR"

echo "[Done] Step 5"

echo ""
echo "========== Pipeline Completed Successfully =========="
echo "Output video: $OUTPUT_DIR/wan/rendered_${NAME}.mp4"
