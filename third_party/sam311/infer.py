import torch
import numpy as np
from PIL import Image
import os

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# ------------------------------------------------
# Function: keep original color in mask, background transparent
# ------------------------------------------------
def save_original_color_mask(mask, image, save_path):
    """
    mask: (H, W) numpy array (0/1)
    image: PIL RGB image
    Output: RGBA PNG (masked area = original image, others = transparent)
    """
    mask = (mask > 0.5).astype(np.uint8)

    img_rgba = np.array(image.convert("RGBA")).copy()

    # Background transparent
    img_rgba[mask == 0, 3] = 0
    img_rgba[mask == 1, 3] = 255

    Image.fromarray(img_rgba).save(save_path)
    print("Saved:", save_path)

# ------------------------------------------------
# Load model
# ------------------------------------------------
model = build_sam3_image_model()
processor = Sam3Processor(model)


# ------------------------------------------------
# Load image
# ------------------------------------------------
image_path = "/gemini/platform/public/aigc/zx/sam-3d-objects/DiffSynth-Studio/multi_scene/plush_toys_group_front/plush_toys_group_front.png"
image = Image.open(image_path).convert("RGB")

# ------------------------------------------------
# Inference
# ------------------------------------------------
inference_state = processor.set_image(image)
output = processor.set_text_prompt(
    state=inference_state,
    prompt="teddy"
)


# "Sphere, circle, Cube, Cylinder, Cone"

masks = output["masks"]  # shape [N, 1, H, W]

# import pdb;pdb.set_trace()

for i, m in enumerate(masks):
    mask_2d = m.squeeze().cpu().numpy()  # (H, W)
    save_path = f"/gemini/platform/public/aigc/zx/sam-3d-objects/DiffSynth-Studio/multi_scene/plush_toys_group_front/{i}.png"
    save_original_color_mask(mask_2d, image, save_path)
