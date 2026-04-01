"""Auto-generate config.yaml (materials + force fields + surface) using Qwen2.5-VL.

Workflow:
  1. Feed the scene image + per-object mask crops to Qwen2.5-VL-72B-Instruct.
  2. VLM identifies each object's material, surface color, and suggests force fields.
  3. Script writes/updates config.yaml for the scene.

Usage:
  python pipeline/auto_config.py --scene_dir data/fluid/sandhouse \
      --model_path /path/to/Qwen2.5-VL-72B-Instruct

  # batch mode: process all scenes under a root
  python pipeline/auto_config.py --root_dir data/fluid \
      --model_path /path/to/Qwen2.5-VL-72B-Instruct
"""

import argparse
import json
import os
import re
import logging
from pathlib import Path

import numpy as np
import yaml
import torch
from PIL import Image

from utils.physics import (
    MATERIAL_DEFAULTS,
    VALID_MATERIALS,
    SURFACE_DEFAULTS as _SURFACE_DEFAULTS,
    VALID_FORCE_TYPES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ── VLM Prompt ──────────────────────────────────────────────────────────

def build_material_prompt(n_objects: int) -> str:
    """Build the VLM prompt for material + force field + surface color identification."""
    return (
        "You are an expert physics material analyst and simulation designer. "
        "I will show you a scene image followed by cropped images of individual objects.\n\n"
        "## Task A: Object Materials\n"
        "For each object (numbered starting from 0), identify:\n"
        "1. What the object is (e.g., sand castle, rubber duck, honey, glass ball)\n"
        "2. Best-matching physics material type for INTERESTING DEFORMABLE simulation.\n"
        "   IMPORTANT: We want diverse, visually interesting fluid/soft-body dynamics.\n"
        "   Prefer non-rigid materials — choose the MOST DEFORMABLE plausible interpretation.\n"
        "   Available material types (Rigid + MPM + PBD):\n\n"
        "   MPM materials (particle-based, great for fluids/deformation):\n"
        '   - "mpm_liquid": liquids, viscous fluids, molten substances (water, honey, syrup, oil, molten glass, lava, paint). Params: E, nu, rho, viscous(bool)\n'
        '   - "mpm_elastoplastic": permanent deformation (clay, dough, cream, ice cream, wax, putty). Params: E, nu, rho, use_von_mises(bool), von_mises_yield_stress\n'
        '   - "mpm_elastic": elastic deformable (rubber, jelly, foam, soft toys, gel, bouncy balls, gummy). Params: E, nu, rho, model("corotation"/"neohooken")\n'
        '   - "mpm_sand": granular (sand, gravel, powder, soil, sugar, salt, beads). Params: E, nu, rho, friction_angle\n'
        '   - "mpm_snow": snow/ice that hardens when compressed. Params: E, nu, rho, yield_lower, yield_higher\n'
        '   - "mpm_muscle": muscle-like active material. Params: E, nu, rho, model, n_groups\n\n'
        "   PBD materials (position-based dynamics):\n"
        '   - "pbd_elastic": 3D soft body (sponge, tofu, soft cube). Params: rho, static_friction, kinetic_friction, stretch/bending/volume_compliance, stretch/bending/volume_relaxation\n'
        '   - "pbd_cloth": 2D thin sheet (cloth, fabric, paper, flag, leaf). Params: rho(kg/m²), static_friction, kinetic_friction, stretch/bending_compliance, stretch/bending_relaxation, air_resistance\n'
        '   - "pbd_liquid": position-based fluid. Params: rho, density_relaxation, viscosity_relaxation\n'
        '   - "pbd_particle": free particles (sparks, debris, confetti). Params: rho\n\n'
        "   Rigid (use sparingly):\n"
        '   - "rigid": ONLY for truly immovable structural objects (ground, wall, table, heavy platform). Params: rho, friction\n'
        "     Do NOT use rigid for small objects, toys, balls, or anything that could plausibly deform or flow.\n"
        "     Example: a glass ball should be mpm_elastic (bouncy), NOT rigid.\n"
        "     Example: a wooden toy should be mpm_elastoplastic, NOT rigid.\n\n"
        "3. material_params: provide relevant params from the list above.\n"
        "   Guide for E (Young modulus): rubber~1e6, foam~1e4, jelly~1e3, cream~5e3, clay~3e4, glass ball~1e5\n"
        "   Guide for rho (density kg/m³): foam~50, cream~500, rubber~1100, clay~1800, glass~2500\n"
        "   Guide for nu (Poisson): 0.2-0.45 typical, nearly incompressible~0.45\n\n"
        "4. fixed: true ONLY if truly static (ground, wall, table, platform), false otherwise\n"
        "5. surface_color: RGB float [0-1] representing the object's visual color "
        "(required for all non-rigid materials)\n\n"
        "## Task B: Force Fields\n"
        "Based on the scene, suggest 1-3 force fields that would create interesting, "
        "physically plausible dynamics. Available types:\n"
        '  - "constant": uniform force (gravity override, push). Params: direction [x,y,z], strength\n'
        '  - "wind": cylindrical wind. Params: direction, strength, radius, center\n'
        '  - "point": attract/repel from a point. Params: strength, position, falloff_pow\n'
        '  - "drag": viscous resistance. Params: linear, quadratic\n'
        '  - "turbulence": random turbulent force. Params: strength, frequency\n'
        '  - "vortex": swirling force. Params: direction, center, strength_perpendicular\n\n'
        "Choose forces that make the scene visually interesting:\n"
        "  - Sand/granular: wind erosion, collapse under gravity, turbulence\n"
        "  - Liquid/honey: gravity drip, wind push, drag for viscosity\n"
        "  - Elastic/jelly: bouncing, squishing, wind deformation\n"
        "  - Cream/dough: gravity collapse, wind blowing\n"
        "  Each force has start_frame (-1=immediate, or N for delayed activation).\n\n"
        f"There are {n_objects} objects (indexed 0 to {n_objects - 1}).\n"
        "Respond with ONLY a JSON object with two keys:\n"
        "{\n"
        '  "objects": [\n'
        '    {"index": 0, "description": "...", "material": "mpm_sand", "material_params": {}, '
        '"fixed": false, "surface_color": [0.76, 0.70, 0.50]},\n'
        "    ...\n"
        "  ],\n"
        '  "forces": [\n'
        '    {"type": "wind", "direction": [1,0,0], "strength": 2.0, "radius": 2.0, '
        '"center": [0,0,0], "start_frame": -1},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
    )


# ── Crop object from scene using mask ────────────────────────────────────

def crop_object_with_mask(scene_img: Image.Image, mask_img: Image.Image) -> Image.Image:
    """Crop the object region from the scene image using its binary mask."""
    mask_arr = np.array(mask_img.convert("L"))
    # Threshold to binary
    binary = (mask_arr > 127).astype(np.uint8)
    ys, xs = np.where(binary)
    if len(ys) == 0:
        return scene_img  # fallback: return full image

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # Add some padding
    pad = max(10, int(0.1 * max(y1 - y0, x1 - x0)))
    h, w = mask_arr.shape
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)

    scene_arr = np.array(scene_img.convert("RGB"))
    # Resize scene to match mask if needed
    if scene_arr.shape[:2] != mask_arr.shape:
        scene_resized = scene_img.convert("RGB").resize(
            (mask_arr.shape[1], mask_arr.shape[0]), Image.LANCZOS
        )
        scene_arr = np.array(scene_resized)

    crop = scene_arr[y0:y1, x0:x1]
    return Image.fromarray(crop)


# ── VLM Model Wrapper ────────────────────────────────────────────────────

class MaterialVLM:
    """Qwen2.5-VL wrapper for material identification."""

    def __init__(self, model_path: str, device_map="auto", torch_dtype=None):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        try:
            import flash_attn  # noqa
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "eager"
            logger.warning("flash_attn not found, using eager attention")

        logger.info(f"Loading {model_path} (device_map={device_map}, attn={attn_impl})")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
            attn_implementation=attn_impl,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        logger.info("Model loaded.")

    @torch.no_grad()
    def analyze_scene(self, scene_image_path: str, object_crops: list[Image.Image]) -> str:
        """Send scene image + object crops to VLM, return raw text output."""
        from qwen_vl_utils import process_vision_info

        content = []
        # Scene image first
        content.append({"type": "image", "image": f"file://{os.path.abspath(scene_image_path)}"})
        content.append({"type": "text", "text": "Above is the full scene image. Below are the individual object crops:\n"})

        # Object crops
        for i, crop in enumerate(object_crops):
            # Save crop to temp file for Qwen VL processing
            tmp_path = f"/tmp/_vlm_crop_{i}.png"
            crop.save(tmp_path)
            content.append({"type": "image", "image": f"file://{tmp_path}"})
            content.append({"type": "text", "text": f"(Object {i})\n"})

        # Prompt
        content.append({"type": "text", "text": build_material_prompt(len(object_crops))})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Cleanup temp files
        for i in range(len(object_crops)):
            tmp_path = f"/tmp/_vlm_crop_{i}.png"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return output


# ── Parse VLM output ─────────────────────────────────────────────────────

def parse_vlm_output(text: str, n_objects: int) -> tuple[list[dict], list[dict]]:
    """Parse VLM JSON output into (object_configs, force_configs).

    Accepts two formats:
      - New: {"objects": [...], "forces": [...]}
      - Legacy: [...]  (objects only, no forces)
    """
    # Try to find JSON object { ... } first (new format)
    match_obj = re.search(r'\{[\s\S]*"objects"[\s\S]*\}', text, re.DOTALL)
    data_objects = None
    data_forces = []

    if match_obj:
        try:
            data = json.loads(match_obj.group())
            data_objects = data.get("objects", [])
            data_forces = data.get("forces", [])
        except json.JSONDecodeError:
            pass

    # Fallback: try to find a plain JSON array (legacy format)
    if data_objects is None:
        match_arr = re.search(r'\[.*\]', text, re.DOTALL)
        if match_arr:
            try:
                data_objects = json.loads(match_arr.group())
            except json.JSONDecodeError:
                pass

    if data_objects is None:
        logger.warning(f"Could not parse VLM output:\n{text}")
        return [_default_object(i) for i in range(n_objects)], []

    # ── Parse objects ──
    results = []
    for i in range(n_objects):
        entry = None
        for d in data_objects:
            if d.get("index") == i:
                entry = d
                break
        if entry is None and i < len(data_objects):
            entry = data_objects[i]
        if entry is None:
            entry = {}

        mat = entry.get("material", "mpm_elastic")
        if mat not in VALID_MATERIALS:
            logger.warning(f"Object {i}: unknown material '{mat}', falling back to 'mpm_elastic'")
            mat = "mpm_elastic"

        obj = {
            "material": mat,
            "fixed": bool(entry.get("fixed", False)),
            "description": entry.get("description", ""),
        }

        # Surface color
        sc = entry.get("surface_color")
        if sc and isinstance(sc, list) and len(sc) == 3:
            obj["surface_color"] = [float(v) for v in sc]
        elif mat in _SURFACE_DEFAULTS:
            obj["surface_color"] = list(_SURFACE_DEFAULTS[mat])

        # Merge material_params (all non-rigid types support params)
        params = entry.get("material_params", {})
        if params and mat != "rigid":
            defaults = MATERIAL_DEFAULTS.get(mat, {})
            merged = {}
            for k, v in defaults.items():
                raw = params.get(k, v)
                # Keep booleans and strings as-is
                if isinstance(v, bool):
                    merged[k] = bool(raw)
                elif isinstance(v, str):
                    merged[k] = str(raw)
                elif isinstance(v, int) and not isinstance(v, bool):
                    merged[k] = int(raw)
                else:
                    merged[k] = float(raw)
            obj["material_params"] = merged

        results.append(obj)

    # ── Parse forces ──
    valid_forces = []
    for fc in data_forces:
        ftype = fc.get("type", "")
        if ftype not in VALID_FORCE_TYPES:
            logger.warning(f"Ignoring unknown force type: {ftype}")
            continue
        valid_forces.append(fc)

    return results, valid_forces


def _default_object(idx: int) -> dict:
    return {"material": "mpm_elastic", "fixed": False, "description": f"object_{idx}",
            "surface_color": [0.80, 0.80, 0.80]}


# ── Generate config.yaml ─────────────────────────────────────────────────

def generate_config(scene_dir: str, vlm_objects: list[dict],
                    vlm_forces: list[dict] = None,
                    force_overwrite: bool = False) -> str:
    """Generate or update config.yaml for a scene."""
    config_path = os.path.join(scene_dir, "config.yaml")
    if vlm_forces is None:
        vlm_forces = []

    # Base simulation config
    cfg = {
        "simulation": {
            "max_retries": 8,
            "dist_thresh": 0.01,
            "cos_thresh": 0.8,
            "low_p": 5,
            "n_steps": 300,
            "fps": 60,
            "camera_mv": 0,
        },
        "objects": {},
    }

    # If config exists, load it to preserve simulation settings
    if os.path.exists(config_path) and not force_overwrite:
        with open(config_path, 'r', encoding='utf-8') as f:
            existing = yaml.safe_load(f) or {}
        if 'simulation' in existing:
            cfg['simulation'] = existing['simulation']

    # Build objects section from VLM results
    for i, obj in enumerate(vlm_objects):
        entry = {
            "material": obj["material"],
            "x_off": 0.0,
            "y_off": 0.0,
            "z_off": 0.0,
            "fixed": obj["fixed"],
            "start_frame": 0,
            "velocity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if obj["material"] == "rigid" else [0.0, 0.0, 0.0],
        }
        if "material_params" in obj:
            entry["material_params"] = obj["material_params"]
        if "surface_color" in obj:
            entry["surface_color"] = obj["surface_color"]
            entry["vis_mode"] = "particle"

        cfg["objects"][i] = entry

    # Write config in project-consistent style (inline lists)
    with open(config_path, 'w', encoding='utf-8') as f:
        scene_name = os.path.basename(scene_dir)
        f.write(f"# Auto-generated config for scene: {scene_name}\n")
        f.write("# Material assignments by Qwen2.5-VL analysis\n")

        for i, obj in enumerate(vlm_objects):
            desc = obj.get("description", "")
            if desc:
                f.write(f"# object_{i}: {desc} -> {obj['material']}\n")
        f.write("\n")

        # Simulation section
        f.write("simulation:\n")
        for k, v in cfg["simulation"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        # Objects section
        f.write("objects:\n")
        for idx in sorted(cfg["objects"].keys()):
            entry = cfg["objects"][idx]
            f.write(f"  {idx}:\n")
            f.write(f"    material: \"{entry['material']}\"\n")
            if "material_params" in entry:
                f.write(f"    material_params:\n")
                for pk, pv in entry["material_params"].items():
                    f.write(f"      {pk}: {pv}\n")
            if "surface_color" in entry:
                sc = entry["surface_color"]
                f.write(f"    surface_color: [{sc[0]}, {sc[1]}, {sc[2]}]\n")
                f.write(f"    vis_mode: \"{entry.get('vis_mode', 'particle')}\"\n")
            f.write(f"    x_off: {entry.get('x_off', 0.0)}\n")
            f.write(f"    y_off: {entry.get('y_off', 0.0)}\n")
            f.write(f"    z_off: {entry.get('z_off', 0.0)}\n")
            f.write(f"    fixed: {'true' if entry.get('fixed') else 'false'}\n")
            f.write(f"    start_frame: {entry.get('start_frame', 0)}\n")
            vel = entry.get("velocity", [0.0, 0.0, 0.0])
            vel_str = "[" + ", ".join(str(v) for v in vel) + "]"
            f.write(f"    velocity: {vel_str}\n")
            f.write("\n")

        # Forces section
        if vlm_forces:
            f.write("forces:\n")
            for fc in vlm_forces:
                ftype = fc.get("type", "constant")
                f.write(f"  - type: \"{ftype}\"\n")
                for k, v in fc.items():
                    if k == "type":
                        continue
                    if isinstance(v, list):
                        v_str = "[" + ", ".join(str(x) for x in v) + "]"
                        f.write(f"    {k}: {v_str}\n")
                    elif isinstance(v, float):
                        f.write(f"    {k}: {v}\n")
                    elif isinstance(v, bool):
                        f.write(f"    {k}: {'true' if v else 'false'}\n")
                    else:
                        f.write(f"    {k}: {v}\n")
                f.write("\n")

    logger.info(f"Config written to {config_path}")
    return config_path


# ── Main pipeline ─────────────────────────────────────────────────────────

def auto_config_scene(scene_dir: str, vlm: "MaterialVLM", force_overwrite: bool = False) -> str:
    """Run auto-config for a single scene directory."""
    scene_dir = os.path.abspath(scene_dir)
    scene_name = os.path.basename(scene_dir)
    image_path = os.path.join(scene_dir, f"{scene_name}.png")
    mask_dir = os.path.join(scene_dir, "rgba_masks")
    sam3d_dir = os.path.join(scene_dir, "sam3d")

    if not os.path.isfile(image_path):
        logger.error(f"Scene image not found: {image_path}")
        return ""

    # Determine number of objects from sam3d or masks
    n_objects = 0
    if os.path.isdir(sam3d_dir):
        n_objects = len([f for f in os.listdir(sam3d_dir) if f.startswith("object_")])
    elif os.path.isdir(mask_dir):
        n_objects = len([f for f in os.listdir(mask_dir)
                         if re.match(r'^\d+\.png$', f)])

    if n_objects == 0:
        logger.warning(f"No objects found in {scene_dir}, generating minimal config")
        n_objects = 1

    logger.info(f"Scene '{scene_name}': {n_objects} objects detected")

    # Load scene image
    scene_img = Image.open(image_path).convert("RGB")

    # Crop each object using masks
    object_crops = []
    for i in range(n_objects):
        mask_path = os.path.join(mask_dir, f"{i}.png")
        if os.path.isfile(mask_path):
            mask_img = Image.open(mask_path)
            crop = crop_object_with_mask(scene_img, mask_img)
        else:
            crop = scene_img  # fallback to full scene
        object_crops.append(crop)

    # Run VLM analysis
    logger.info(f"Running VLM analysis on {scene_name} ({n_objects} objects)...")
    raw_output = vlm.analyze_scene(image_path, object_crops)
    logger.info(f"VLM raw output:\n{raw_output}")

    # Parse VLM output
    vlm_objects, vlm_forces = parse_vlm_output(raw_output, n_objects)

    for i, obj in enumerate(vlm_objects):
        sc_info = f", surface={obj['surface_color']}" if "surface_color" in obj else ""
        logger.info(f"  object_{i}: {obj.get('description', '?')} -> {obj['material']} "
                     f"(fixed={obj['fixed']}{sc_info})")
    for fc in vlm_forces:
        logger.info(f"  force: {fc.get('type')} strength={fc.get('strength', '?')} "
                     f"start_frame={fc.get('start_frame', -1)}")

    # Generate config.yaml
    config_path = generate_config(scene_dir, vlm_objects, vlm_forces, force_overwrite)
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Auto-generate config.yaml using Qwen2.5-VL")
    parser.add_argument("--scene_dir", type=str, default=None,
                        help="Path to a single scene directory")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Process all scene dirs under this root")
    parser.add_argument("--model_path", type=str,
                        default="/gemini/platform/public/aigc/cyb/zx_plus/Workshop/models/Qwen2.5-VL-72B-Instruct",
                        help="Path to Qwen2.5-VL model")
    parser.add_argument("--force_overwrite", action="store_true",
                        help="Overwrite existing config.yaml completely (default: preserve simulation settings)")
    parser.add_argument("--device_map", type=str, default="auto")

    args = parser.parse_args()

    if not args.scene_dir and not args.root_dir:
        parser.error("Must specify --scene_dir or --root_dir")

    # Load VLM model once
    vlm = MaterialVLM(model_path=args.model_path, device_map=args.device_map)

    if args.scene_dir:
        auto_config_scene(args.scene_dir, vlm, force_overwrite=args.force_overwrite)
    elif args.root_dir:
        # Find all scene directories (contain {name}.png)
        root = args.root_dir
        for entry in sorted(os.listdir(root)):
            scene_dir = os.path.join(root, entry)
            if not os.path.isdir(scene_dir):
                continue
            scene_img = os.path.join(scene_dir, f"{entry}.png")
            if os.path.isfile(scene_img):
                auto_config_scene(scene_dir, vlm, force_overwrite=args.force_overwrite)

    torch.cuda.empty_cache()
    logger.info("Done.")


if __name__ == "__main__":
    main()
