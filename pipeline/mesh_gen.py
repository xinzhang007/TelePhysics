import os
import sys
import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix
import argparse

from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from sam3d_objects.inference import Inference, load_image, load_masks


_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

_R_ZUP_TO_NEGYUP = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0,-1, 0],
], dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root directory that contains all scenes"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="hf",
        help="Model tag (checkpoint folder)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--scenes_name",
        type=str,
        nargs="*",
        default=[],  # Empty list means process all scenes
        help="List of specific scene names to process"
    )

    return parser.parse_args()


def process_scene(scene_dir, scene_name, inference, seed):
    print(f"\n====== Processing scene: {scene_name} ======")

    image_path = os.path.join(scene_dir, f"{scene_name}.png")
    mask_dir = os.path.join(scene_dir, "rgba_masks")

    if not os.path.exists(image_path):
        print(f"[Skip] image not found: {image_path}")
        return

    # if not os.path.isdir(mask_dir):
    #     print(f"[Skip] mask dir not found: {mask_dir}")
    #     return

    # Load image & masks
    image = load_image(image_path)
    masks = load_masks(mask_dir, extension=".png")

    # Run inference on each mask
    outputs = [
        inference(image, mask, seed=seed)
        for mask in masks
    ]
    
    
    # Process & export meshes
    for i, output in enumerate(outputs):
        mesh = output["glb"]
        
        if mesh is None:
            continue

        # Convert coords Y-up <-> Z-up
        vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)

        R_l2c = quaternion_to_matrix(output["rotation"])

        l2c_transform = compose_transform(
            scale=output["scale"],
            rotation=R_l2c,
            translation=output["translation"],
        )

        vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
      
        # mesh.vertices = vertices.squeeze(0).cpu().numpy() @ _R_ZUP_TO_YUP

        mesh.vertices = vertices.squeeze(0).cpu().numpy() @ _R_ZUP_TO_NEGYUP
         
        # Save output directory
        save_dir = os.path.join(scene_dir, "sam3d", f"object_{i}")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "mesh.obj")
        mesh.export(save_path)

        print(f"[OK] Saved mesh: {save_path}")


def main():
    args = parse_args()

    # Initialize inference pipeline
    config_path = f"models/SAM3D/{args.tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    scene_names = sorted([
        d for d in os.listdir(args.root_dir)
        if os.path.isdir(os.path.join(args.root_dir, d))
    ])

    # Filter scene names if specified
    if args.scenes_name:
        scene_names = [scene for scene in scene_names if scene in args.scenes_name]

    print("Found scenes:")
    for n in scene_names:
        print(" -", n)
    
    # Process each scene
    for scene_name in scene_names:

        scene_dir = os.path.join(args.root_dir, scene_name)
        # save_dir = os.path.join(scene_dir, "sam3d")
        # if os.path.exists(save_dir):
        #     print("mesh existing")
        # else:
        
        process_scene(scene_dir, scene_name, inference, args.seed)

    print("\n====== All specified scenes processed ======")


if __name__ == "__main__":
    main()
