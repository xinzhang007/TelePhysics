import cv2
import numpy as np
from PIL import Image


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)



def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask




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