"""
Camera Alignment Module — Solves the input-image-to-simulation alignment problem.

Root cause: normalize_scene_to_ground applies R @ (v - centroid) to meshes,
destroying the original camera-scene correspondence. The current pipeline
then tries random search to recover it — fundamentally broken.

Solution (3-tier):
  1. DA3 Camera Prior: Depth Anything 3 estimates (intrinsics, extrinsics) from
     the input image, giving an analytical camera pose.
  2. Coordinate Bookkeeping: Track the exact (R, t, z_lift) applied during
     normalize_scene_to_ground, then transform the DA3 camera accordingly.
  3. PnP Refinement: Use 2D mask centroids + 3D mesh centroids for Perspective-n-Point
     solve as fine-tuning — closed-form, no random search.

The key insight: the camera that took the input photo is exactly the camera
we need in Genesis. We just need to transform it through the same pipeline
that the meshes went through.
"""

import os
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# 1. Track scene normalization transform (so we can apply it to the camera)
# ===========================================================================

def save_scene_transform(
    output_path: str,
    R_norm: np.ndarray,
    centroid: np.ndarray,
    z_lift: float,
):
    """
    Save the exact transform applied during normalize_scene_to_ground.

    The mesh transform is:  v_new = R_norm @ (v_orig - centroid);  v_new[2] -= z_lift

    To transform a camera through the same pipeline:
      pos_new  = R_norm @ (pos_orig - centroid);  pos_new[2] -= z_lift
      lookat_new = R_norm @ lookat_dir_orig  (direction, no translation)
    """
    np.savez(
        output_path,
        R_norm=R_norm,
        centroid=centroid,
        z_lift=z_lift,
    )


def load_scene_transform(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load saved (R_norm, centroid, z_lift)."""
    data = np.load(path)
    return data["R_norm"], data["centroid"], float(data["z_lift"])


# ===========================================================================
# 2. DA3 Camera Prior — extract camera pose from input image
# ===========================================================================

_da3_model_cache = None


def _get_da3_model(device: str = "cuda"):
    """Lazy-load DA3 model (cached singleton)."""
    global _da3_model_cache
    if _da3_model_cache is not None:
        return _da3_model_cache
    from depth_anything_3.api import DepthAnything3
    import torch
    model = DepthAnything3(model_name="da3-large")
    model = model.to(device).eval()
    _da3_model_cache = model
    return model


def estimate_camera_da3(
    image_path: str,
    da3_model=None,
    device: str = "cuda",
) -> Dict:
    """
    Use Depth Anything 3 to estimate camera pose + depth from a single image.

    DA3 outputs:
      - depth: (N, H, W) relative depth map
      - extrinsics: (N, 3, 4) world-to-camera [R|t]
      - intrinsics: (N, 3, 3) — may have placeholder fx for single images

    We extract the camera rotation (reliable) and use heuristic FOV when
    the intrinsics are clearly wrong (fx > 1e6).

    Returns:
        dict with 'intrinsics', 'extrinsics', 'fov_deg', 'position',
        'lookat_dir', 'depth'.
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    try:
        model = da3_model if da3_model is not None else _get_da3_model(device)
        prediction = model.inference(image=[image_path])
    except Exception as e:
        print(f"[DA3] Inference failed: {e}")
        return _heuristic_camera(H, W)

    # Extract raw outputs
    depth = prediction.depth[0]  # (proc_H, proc_W)
    proc_H, proc_W = depth.shape

    # Extrinsics: DA3 returns (N, 3, 4), expand to (4, 4)
    ext_raw = prediction.extrinsics[0]  # (3, 4) or (4, 4)
    if ext_raw.shape[0] == 3:
        extrinsics = np.eye(4, dtype=np.float64)
        extrinsics[:3, :] = ext_raw.astype(np.float64)
    else:
        extrinsics = ext_raw.astype(np.float64)

    R_w2c = extrinsics[:3, :3]
    t_w2c = extrinsics[:3, 3]
    position = -R_w2c.T @ t_w2c
    lookat_dir = R_w2c.T @ np.array([0.0, 0.0, 1.0])

    # Intrinsics: DA3 may return placeholder values for single images
    K_raw = prediction.intrinsics[0].astype(np.float64) if prediction.intrinsics is not None else None
    fx_raw = K_raw[0, 0] if K_raw is not None else 0.0

    # Sanity check: for a proc_W=504 image, reasonable fx is ~250-1000.
    # If fx > 1e6, it's a placeholder — use heuristic FOV instead.
    if fx_raw > 1e6 or fx_raw < 1.0:
        fov_deg = 60.0  # default heuristic
        fx = (proc_W / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
        intrinsics = np.array([
            [fx, 0, proc_W / 2.0],
            [0, fx, proc_H / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)
        print(f"[DA3] Intrinsics placeholder detected (fx={fx_raw:.0f}), using heuristic FOV={fov_deg}°")
    else:
        intrinsics = K_raw
        fov_deg = float(np.degrees(2.0 * np.arctan(proc_W / (2.0 * fx_raw))))

    print(f"[DA3] pos={position}, lookat={lookat_dir}, fov={fov_deg:.1f}°, "
          f"depth=[{depth.min():.3f}, {depth.max():.3f}]")

    return {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "fov_deg": float(fov_deg),
        "position": position,
        "lookat_dir": lookat_dir,
        "depth": depth,
    }


def _heuristic_camera(H: int, W: int) -> Dict:
    """Fallback: assume frontal camera looking at -Y with 60° FOV."""
    fov = 60.0
    fx = (W / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    intrinsics = np.array([
        [fx, 0, W / 2.0],
        [0, fx, H / 2.0],
        [0, 0, 1],
    ])
    extrinsics = np.eye(4)
    return {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "fov_deg": fov,
        "position": np.array([0.0, 0.0, 0.0]),
        "lookat_dir": np.array([0.0, -1.0, 0.0]),
        "depth": None,
    }


# ===========================================================================
# 3. Transform DA3 camera through the scene normalization pipeline
# ===========================================================================

def transform_camera_through_normalization(
    cam_position: np.ndarray,
    cam_lookat_dir: np.ndarray,
    R_norm: np.ndarray,
    centroid: np.ndarray,
    z_lift: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the SAME transform that normalize_scene_to_ground applied to meshes,
    to the camera pose. This keeps camera-scene correspondence exact.

    Mesh transform was:
        v_new = R_norm @ (v_orig - centroid)
        v_new[2] -= z_lift

    So camera must undergo the same:
        pos_new = R_norm @ (pos_orig - centroid)
        pos_new[2] -= z_lift
        lookat_dir_new = R_norm @ lookat_dir_orig  (rotation only, no translation)

    Args:
        cam_position: (3,) camera position in original mesh frame.
        cam_lookat_dir: (3,) camera look-at direction (unit vector).
        R_norm: (3,3) rotation from normalize_scene_to_ground.
        centroid: (3,) centroid subtracted during normalization.
        z_lift: scalar z offset applied after rotation.

    Returns:
        (new_position, new_lookat_point)
    """
    pos = np.asarray(cam_position, dtype=np.float64)
    look_dir = np.asarray(cam_lookat_dir, dtype=np.float64)

    # Apply same transform as meshes
    new_pos = R_norm @ (pos - centroid)
    new_pos[2] -= z_lift

    # Direction is only rotated, not translated
    new_dir = R_norm @ look_dir
    new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-12)

    # Compute a lookat point (position + direction)
    new_lookat = new_pos + new_dir

    return new_pos, new_lookat


# ===========================================================================
# 4. PnP refinement — use 2D-3D correspondences for precise alignment
# ===========================================================================

def refine_camera_pnp(
    mask_dir: str,
    mesh_dir: str,
    intrinsics: np.ndarray,
    initial_pos: np.ndarray,
    initial_lookat: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    PnP-based camera refinement using 2D mask centroids ↔ 3D mesh centroids.

    This is a closed-form solution (no random search) that uses:
      - 2D: centroid of each object's mask in the input image
      - 3D: centroid of each object's normalized mesh

    Much more reliable than pixel-level optimization because:
      1. It's a direct geometric solve, not iterative search
      2. Each object gives one high-confidence correspondence
      3. OpenCV's solvePnP handles the math

    Args:
        mask_dir: directory with per-object RGBA masks (0.png, 1.png, ...).
        mesh_dir: directory with object_*/mesh_norm.obj files.
        intrinsics: (3,3) camera intrinsic matrix.
        initial_pos: (3,) initial camera position (from DA3 or heuristic).
        initial_lookat: (3,) initial lookat point.
        image_size: (H, W).

    Returns:
        (refined_pos, refined_lookat, fov_deg)
    """
    import trimesh

    H, W = image_size
    points_3d = []
    points_2d = []

    # Collect 2D centroids from masks (only numeric filenames like 0.png, 1.png)
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.endswith('.png') and os.path.splitext(f)[0].isdigit()
    ])

    for mask_file in mask_files:
        idx = int(os.path.splitext(mask_file)[0])
        mask_path = os.path.join(mask_dir, mask_file)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            continue

        # Extract alpha or convert to binary
        if mask_img.ndim == 3 and mask_img.shape[2] == 4:
            alpha = mask_img[:, :, 3]
        elif mask_img.ndim == 3:
            alpha = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        else:
            alpha = mask_img

        ys, xs = np.where(alpha > 127)
        if len(xs) < 10:
            continue

        # 2D centroid in image coordinates
        cx_2d = xs.mean()
        cy_2d = ys.mean()

        # Scale to rendering resolution
        orig_h, orig_w = alpha.shape[:2]
        cx_2d = cx_2d * W / orig_w
        cy_2d = cy_2d * H / orig_h

        # 3D centroid from normalized mesh
        mesh_path = os.path.join(mesh_dir, f"object_{idx}", "mesh_norm.obj")
        if not os.path.isfile(mesh_path):
            continue

        mesh = trimesh.load(mesh_path, force='mesh')
        if mesh.is_empty:
            continue

        centroid_3d = mesh.vertices.mean(axis=0)

        points_2d.append([cx_2d, cy_2d])
        points_3d.append(centroid_3d)

    if len(points_3d) < 3:
        print(f"[PnP] Only {len(points_3d)} correspondences, need ≥3. Using initial guess.")
        return initial_pos, initial_lookat, _fov_from_intrinsics(intrinsics, W)

    points_3d = np.array(points_3d, dtype=np.float64)
    points_2d = np.array(points_2d, dtype=np.float64)

    # Compute scene extent for sanity checking PnP results
    scene_extent = np.ptp(points_3d, axis=0).max()  # max range across x/y/z
    max_reasonable_dist = max(scene_extent * 20.0, 10.0)  # camera shouldn't be >20x scene extent away

    K_mat = intrinsics[:3, :3].astype(np.float64)
    fov_deg = _fov_from_intrinsics(intrinsics, W)

    # Try multiple PnP methods, pick the best one that passes sanity check
    candidates = []

    # Method 1: SQPNP (no initial guess needed, robust for few points)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d, K_mat, distCoeffs=None,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if ok:
            pos, lookat = _rvec_tvec_to_pos_lookat(rvec, tvec)
            candidates.append(("SQPNP", pos, lookat))
    except Exception:
        pass

    # Method 2: Iterative with initial guess
    try:
        rvec_init, tvec_init = _pos_lookat_to_rvec_tvec(initial_pos, initial_lookat)
        ok, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d, K_mat, distCoeffs=None,
            rvec=rvec_init, tvec=tvec_init,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            pos, lookat = _rvec_tvec_to_pos_lookat(rvec, tvec)
            candidates.append(("ITERATIVE", pos, lookat))
    except Exception:
        pass

    # Method 3: IPPE (for 4+ coplanar-ish points)
    if len(points_3d) >= 4:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                points_3d, points_2d, K_mat, distCoeffs=None,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if ok:
                pos, lookat = _rvec_tvec_to_pos_lookat(rvec, tvec)
                candidates.append(("IPPE", pos, lookat))
        except Exception:
            pass

    # Filter candidates by sanity check: distance + viewing direction consistency
    scene_center = points_3d.mean(axis=0)
    # Reference viewing direction from initial guess (DA3 or heuristic)
    init_view_dir = np.asarray(initial_lookat) - np.asarray(initial_pos)
    init_view_dir = init_view_dir / (np.linalg.norm(init_view_dir) + 1e-12)

    valid_candidates = []
    for name, pos, lookat in candidates:
        dist = np.linalg.norm(pos - scene_center)
        if dist >= max_reasonable_dist:
            print(f"[PnP] {name}: pos={pos}, dist_from_scene={dist:.1f}m (REJECTED, >{max_reasonable_dist:.1f}m)")
            continue

        # Direction consistency check: reject 180° flipped solutions
        # PnP with few points often produces a mirror solution where the camera
        # is on the opposite side of the scene looking back.
        cand_view_dir = np.asarray(lookat) - np.asarray(pos)
        cand_view_dir = cand_view_dir / (np.linalg.norm(cand_view_dir) + 1e-12)
        dot = float(np.dot(init_view_dir, cand_view_dir))

        if dot < -0.3:
            # Camera is looking in roughly the opposite direction — likely a flipped PnP solution
            print(f"[PnP] {name}: pos={pos}, view_dot={dot:.2f} (REJECTED, flipped direction)")
            continue

        valid_candidates.append((name, pos, lookat, dist))
        print(f"[PnP] {name}: pos={pos}, dist={dist:.3f}m, view_dot={dot:.2f} (valid)")

    if not valid_candidates:
        # Compute reprojection error for initial guess as a baseline
        reproj_err = _reprojection_error(points_3d, points_2d, initial_pos, initial_lookat, K_mat)
        print(f"[PnP] All PnP solutions failed sanity check. Using initial guess (reproj_err={reproj_err:.1f}px).")
        return initial_pos, initial_lookat, fov_deg

    # Pick the candidate with smallest reprojection error
    best = None
    best_err = float('inf')
    for name, pos, lookat, dist in valid_candidates:
        err = _reprojection_error(points_3d, points_2d, pos, lookat, K_mat)
        print(f"[PnP] {name}: reprojection_error={err:.1f}px")
        if err < best_err:
            best_err = err
            best = (name, pos, lookat)

    name, cam_pos, cam_lookat = best
    print(f"[PnP] Selected: {name}, pos={cam_pos}, fov={fov_deg:.1f}°, reproj_err={best_err:.1f}px")

    return cam_pos, cam_lookat, fov_deg


# ===========================================================================
# 5. Mask-IoU based coarse alignment (fast hemisphere selection)
# ===========================================================================

def mask_iou_search(
    cam,
    bg: np.ndarray,
    target_mask: np.ndarray,
    center: np.ndarray,
    radius: float,
    n_samples: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Fast coarse search using mask IoU instead of pixel MSE.

    Mask IoU is much more robust to texture differences and only cares about
    object silhouette alignment. This finds the correct camera hemisphere
    before fine-tuning.

    Args:
        cam: Genesis camera.
        bg: background image.
        target_mask: (H,W) binary mask of objects in input image.
        center: (3,) center point to sample around.
        radius: sampling radius.
        n_samples: number of random camera positions to try.
        seed: random seed.

    Returns:
        (best_position, best_iou)
    """
    from utils.utils_sim import render_object_only

    rng = np.random.default_rng(seed)
    target_binary = (target_mask > 0.5).astype(np.float32)
    target_area = target_binary.sum()

    if target_area < 10:
        return center, 0.0

    best_pos = center.copy()
    best_iou = -1.0

    for _ in range(n_samples):
        # Sample on a sphere around center
        offset = rng.standard_normal(3)
        offset = offset / (np.linalg.norm(offset) + 1e-8) * radius
        # Add some z bias (cameras are usually above the scene)
        offset[2] = abs(offset[2])
        pos = center + offset

        cam.set_pose(pos=pos.tolist())
        _, obj_mask = render_object_only(cam, bg)

        # Compute IoU
        render_binary = (obj_mask.squeeze() > 0.5).astype(np.float32)
        intersection = (target_binary * render_binary).sum()
        union = target_binary.sum() + render_binary.sum() - intersection

        iou = intersection / (union + 1e-6)

        if iou > best_iou:
            best_iou = iou
            best_pos = pos.copy()

    return best_pos, float(best_iou)


# ===========================================================================
# 6. Full alignment pipeline
# ===========================================================================

def align_camera(
    image_path: str,
    mask_dir: str,
    mesh_dir: str,
    scene_transform_path: Optional[str] = None,
    image_size: Tuple[int, int] = (880, 880),
    device: str = "cuda",
) -> Dict:
    """
    Full camera alignment pipeline. Returns camera parameters for Genesis.

    Strategy (in order of preference):
      1. DA3 camera prior → transform through normalization → PnP refine
      2. PnP from mask centroids + mesh centroids (if DA3 unavailable)
      3. Mask-IoU coarse search + pixel refinement (fallback)

    Args:
        image_path: input image path.
        mask_dir: per-object mask directory.
        mesh_dir: normalized mesh directory (sam3d/).
        scene_transform_path: path to scene_transform.npz (from normalize step).
        image_size: (H, W) rendering resolution.
        device: torch device.

    Returns:
        dict with 'position', 'lookat', 'fov', 'method' (which strategy succeeded).
    """
    import trimesh

    H, W = image_size

    # --- Compute scene center from normalized meshes (always useful) ---
    all_centers = []
    if os.path.isdir(mesh_dir):
        for folder in sorted(os.listdir(mesh_dir)):
            if not folder.startswith("object_"):
                continue
            mp = os.path.join(mesh_dir, folder, "mesh_norm.obj")
            if os.path.isfile(mp):
                m = trimesh.load(mp, force='mesh')
                if not m.is_empty:
                    all_centers.append(m.vertices.mean(axis=0))

    scene_center = np.mean(all_centers, axis=0) if all_centers else np.zeros(3)

    # Default intrinsics (60° FOV)
    fov_default = 60.0
    fx = (W / 2.0) / np.tan(np.deg2rad(fov_default / 2.0))
    K_default = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float64)

    # --- Strategy 1: DA3 camera prior ---
    da3_result = None
    da3_is_real = False  # True only if DA3 actually ran (not heuristic fallback)
    try:
        da3_result = estimate_camera_da3(image_path, device=device)
        # Check if DA3 actually produced a real estimate (not heuristic fallback)
        da3_is_real = da3_result.get("depth") is not None
        if da3_is_real:
            print(f"[CameraAlign] DA3 estimated FOV={da3_result['fov_deg']:.1f}°, "
                  f"pos={da3_result['position']}")
        else:
            print("[CameraAlign] DA3 unavailable, using heuristic camera prior")
    except Exception as e:
        print(f"[CameraAlign] DA3 unavailable: {e}")

    # --- Build initial camera pose ---
    # Priority: DA3 (if real) → scene-center heuristic
    init_fov = fov_default
    K_init = K_default.copy()

    if da3_is_real and da3_result is not None and scene_transform_path and os.path.exists(scene_transform_path):
        # Transform the real DA3 camera through scene normalization
        R_norm, centroid, z_lift = load_scene_transform(scene_transform_path)
        init_pos, init_lookat = transform_camera_through_normalization(
            da3_result["position"],
            da3_result["lookat_dir"],
            R_norm, centroid, z_lift,
        )
        init_fov = da3_result["fov_deg"]
        # Scale DA3 intrinsics to rendering resolution
        intrinsics = da3_result["intrinsics"]
        depth_h, depth_w = da3_result["depth"].shape
        K_init = intrinsics.copy()
        K_init[0, :] *= W / depth_w
        K_init[1, :] *= H / depth_h
        method_base = "da3+transform"
        print(f"[CameraAlign] DA3 camera after normalization: pos={init_pos}, lookat={init_lookat}")
    elif all_centers:
        # Scene-center heuristic: camera at +Y looking at scene center
        # This matches the old transform_camera_with_mesh behavior
        init_pos = scene_center + np.array([0.0, 2.0, 0.5])
        init_lookat = scene_center
        method_base = "heuristic"
        print(f"[CameraAlign] Heuristic camera: pos={init_pos}, lookat={init_lookat}")
    else:
        print("[CameraAlign] No meshes found, using default camera")
        return {
            "position": [0, 0, 0],
            "lookat": [0, -1, 0],
            "fov": fov_default,
            "method": "heuristic",
        }

    # --- Strategy 2: PnP refinement from initial pose ---
    if os.path.isdir(mask_dir) and os.path.isdir(mesh_dir) and len(all_centers) >= 3:
        pos_ref, lookat_ref, fov_ref = refine_camera_pnp(
            mask_dir, mesh_dir, K_init,
            init_pos, init_lookat, image_size,
        )
        # Check if PnP actually changed anything (it returns initial guess if failed)
        pnp_moved = np.linalg.norm(np.array(pos_ref) - np.array(init_pos)) > 1e-6
        method = f"{method_base}+pnp" if pnp_moved else method_base

        return {
            "position": pos_ref.tolist() if hasattr(pos_ref, 'tolist') else list(pos_ref),
            "lookat": lookat_ref.tolist() if hasattr(lookat_ref, 'tolist') else list(lookat_ref),
            "fov": fov_ref,
            "method": method,
        }

    # --- No PnP possible (< 3 objects), use initial pose directly ---
    return {
        "position": init_pos.tolist() if hasattr(init_pos, 'tolist') else list(init_pos),
        "lookat": init_lookat.tolist() if hasattr(init_lookat, 'tolist') else list(init_lookat),
        "fov": init_fov,
        "method": method_base,
    }


# ===========================================================================
# Helpers
# ===========================================================================

def _rvec_tvec_to_pos_lookat(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV rvec/tvec to camera position + lookat point."""
    R_cv, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R_cv.T @ tvec).flatten()
    look_dir = R_cv.T @ np.array([0, 0, 1.0])
    cam_lookat = cam_pos + look_dir
    return cam_pos, cam_lookat


def _reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    cam_pos: np.ndarray,
    cam_lookat: np.ndarray,
    K: np.ndarray,
) -> float:
    """Mean reprojection error (pixels) for a camera pose."""
    rvec, tvec = _pos_lookat_to_rvec_tvec(cam_pos, cam_lookat)
    projected, _ = cv2.projectPoints(
        points_3d, rvec, tvec, K, distCoeffs=None,
    )
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - points_2d, axis=1)))


def _fov_from_intrinsics(K: np.ndarray, W: int) -> float:
    fx = K[0, 0]
    return float(np.degrees(2.0 * np.arctan(W / (2.0 * fx))))


def _pos_lookat_to_rvec_tvec(
    pos: np.ndarray,
    lookat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert camera position + lookat to OpenCV rvec/tvec."""
    pos = np.asarray(pos, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)

    # Camera Z axis = normalized(lookat - pos)
    z = lookat - pos
    z = z / (np.linalg.norm(z) + 1e-12)

    # Camera X axis = z × up (approximate)
    up = np.array([0, 0, 1.0])
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        up = np.array([0, 1, 0.0])
        x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-12)

    # Camera Y axis
    y = np.cross(x, z)

    # R_w2c: world to camera rotation
    R = np.stack([x, y, z], axis=0)  # (3,3), each row is an axis
    t = -R @ pos  # translation

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    return rvec, tvec
