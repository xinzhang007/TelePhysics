# import torch
# import numpy as np
# from PIL import Image
# import os

# from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor

import numpy as np
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt
import torch

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
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = pipeline("mask-generation", model="facebook/sam3", device=device)


# ------------------------------------------------
# Load image
# ------------------------------------------------
image_path = "/gemini/platform/public/aigc/zx/sam-3d-objects/bricks/bricks.png"
image = Image.open(image_path).convert("RGB")

# ------------------------------------------------
# Inference
# ------------------------------------------------
results = generator(image)
masks = results["masks"]
print(f"Found {len(masks)} masks in the image.")


# import pdb;pdb.set_trace()

for i, m in enumerate(masks):
    mask_2d = m.squeeze().cpu().numpy()  # (H, W)
    save_path = f"bricks/{i}.png"
    save_original_color_mask(mask_2d, image, save_path)
