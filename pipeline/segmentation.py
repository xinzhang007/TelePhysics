import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_PROJECT_ROOT, "third_party", "sam311"))
sys.path.append(os.path.join(_PROJECT_ROOT, "third_party", "sam311", "submodules", "lama"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from lama_inpaint import inpaint_img_with_lama
from utils_sam3 import load_img_to_array, save_array_to_img, save_original_color_mask


# ------------------------------------------------
# Config
# ------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj_name", type=str, default="apple")
    parser.add_argument("--text_prompt", type=str, nargs="+", default=["apple"])
    parser.add_argument("--root_dir", type=str, default="data/multi_object")

    args = parser.parse_args()

    args.obj_dir = os.path.join(args.root_dir, args.obj_name)

    return args

# ------------------------------------------------
# Main Pipeline
# ------------------------------------------------
def main(args):

    obj_name     = args.obj_name
    text_prompt  = args.text_prompt
    obj_dir      = args.obj_dir

    # import pdb;pdb.set_trace()

    # Build paths
    image_path       = os.path.join(obj_dir, f"{obj_name}.png")
    output_dir       = obj_dir
    masks_dir        = os.path.join(output_dir, "masks")
    inpaint_dir      = os.path.join(output_dir, "inpaint")
    rgba_masks_dir   = os.path.join(output_dir, "rgba_masks")
    # outpaint_dir     = os.path.join(output_dir, "outpaint")

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(inpaint_dir, exist_ok=True)
    os.makedirs(rgba_masks_dir, exist_ok=True)
    # os.makedirs(outpaint_dir, exist_ok=True)

    # ------------------------------------------------
    # LAMA Config
    # ------------------------------------------------
    lama_config       = os.path.join(_PROJECT_ROOT, "configs", "lama-prediction.yaml")
    lama_ckpt         = os.path.join(_PROJECT_ROOT, "models", "big-lama")
    device            = "cuda" if torch.cuda.is_available() else "cpu"
    dilate_kernel     = 100
    lama_find_shade   = False


    print("\n=== Loading SAM3 Model ===")
    model = build_sam3_image_model(checkpoint_path=os.path.join(_PROJECT_ROOT, "models", "SAM3", "sam3.pt"))
    processor = Sam3Processor(model)
    image_pil = Image.open(image_path).convert("RGB")
  
    state   = processor.set_image(image_pil)
    
    masks_list=[]
    if len(set(text_prompt)) == 1:
        prompt = text_prompt[0]


    for prompt in text_prompt:
        result  = processor.set_text_prompt(state=state, prompt=prompt)
        masks   = result["masks"]
        num_masks = len(masks)

        print(f"Found {len(masks)} masks for prompt: '{prompt}'")
        # if num_masks == 0:
        #     print("No masks found. Skipping inpainting and outpainting steps.")
        #     return
        
        masks_list.append(masks)
    
    masks = torch.cat(masks_list, dim=0)

    print("\n=== Saving RGBA Masks ===")

    out_paths = []

    # 用于叠加所有mask（初始化为全0）
    combined_mask = torch.zeros_like(masks[0])

    for idx, m in enumerate(masks):
        mask_2d = m.squeeze().cpu().numpy()
        out_path = os.path.join(rgba_masks_dir, f"{idx}.png")
        
        save_original_color_mask(mask_2d, image_pil, out_path)
        out_paths.append(out_path)
        print(f"✓ Saved RGBA mask: {out_path}")

        combined_mask = torch.logical_or(combined_mask, m)

    combined_mask_2d = combined_mask.squeeze().cpu().numpy().astype("uint8") * 255
    combined_path = os.path.join(rgba_masks_dir, "combined_binary.png")
    Image.fromarray(combined_mask_2d).save(combined_path)
    print(f"\n✓ Saved combined binary mask: {combined_path}")


    print("\n=== Running Inpainting ===")
    ############################################################################################################################

    img      = load_img_to_array(image_path)
    last_img = img.copy()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _inpaint_single(idx, m):
        """Inpaint a single object mask (for parallel execution)."""
        mask_np = (m.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        out_single = inpaint_img_with_lama(
            img,
            mask_np,
            lama_config,
            lama_ckpt,
            device=device,
            dilation=dilate_kernel,
            find_shade=lama_find_shade,
            out_path=os.path.join(masks_dir, f"mask_{idx}_final.jpg"),
        )
        save_array_to_img(out_single, os.path.join(inpaint_dir, f"inpaint_single_{idx}.png"))
        return idx, mask_np, out_single

    # Run single-object inpainting in parallel
    with ThreadPoolExecutor(max_workers=min(4, len(masks))) as executor:
        futures = {
            executor.submit(_inpaint_single, idx, m): idx
            for idx, m in enumerate(masks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
                print(f"Inpainted single object: {idx}")
            except Exception as e:
                print(f"Inpainting failed for object {idx}: {e}")

    # Cumulative inpainting must remain sequential
    for idx, m in enumerate(masks):
        mask_np = (m.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        print(f"Inpainting cumulative (object {idx})")
        last_img = inpaint_img_with_lama(
            last_img,
            mask_np,
            lama_config,
            lama_ckpt,
            device=device,
            dilation=dilate_kernel,
            find_shade=lama_find_shade,
            out_path=os.path.join(masks_dir, f"mask_{idx}_final_cumu.jpg"),
        )
        save_array_to_img(last_img, os.path.join(inpaint_dir, f"inpaint_cumu_{idx}.png"))


    final_inpaint_path = os.path.join(inpaint_dir, "inpaint_all.png")
    save_array_to_img(last_img, final_inpaint_path)
    print(f"Final inpainted image saved: {final_inpaint_path}")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
