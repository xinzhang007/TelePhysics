import os
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.signal import fftconvolve


# ---------- 相机参数加载 ----------
def load_camera_from_json(cameras_json_path, frame_idx=0):
    """Load camera parameters from cameras.json and convert to Genesis world coords.

    The coordinate conversion matches the ``_R_ZUP_TO_NEGYUP`` transform
    applied in ``mesh_gen.py``.

    Parameters
    ----------
    cameras_json_path : str – path to ``cameras.json``.
    frame_idx : int – which frame to use (default 0).

    Returns
    -------
    dict with keys:
        pos      : tuple  – camera position in Genesis coords
        lookat   : tuple  – camera lookat point
        up       : tuple  – camera up vector
        fwd      : np.ndarray – forward direction vector (Genesis coords, not unit)
        fov      : float  – vertical field of view in degrees
        img_w    : int    – image width  (= 2 * cx)
        img_h    : int    – image height (= 2 * cy)
        fx, fy   : float  – focal lengths from intrinsics
    """
    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)

    K = np.array(cameras_data['intrinsics'][frame_idx])   # 3x3
    E = np.array(cameras_data['extrinsics'][frame_idx])   # 4x4 w2c

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    img_w, img_h = int(2 * cx), int(2 * cy)

    # mesh_gen.py applies _R_ZUP_TO_NEGYUP as the final vertex transform;
    # the same rotation converts camera-space vectors to Genesis world coords.
    _R_CONV = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

    c2w = np.linalg.inv(E)
    cam_pos_w = c2w[:3, 3]
    cam_fwd_w = c2w[:3, 2]          # OpenCV +Z = forward
    cam_up_w  = c2w[:3, 1]           # +Y column

    pos_gs    = tuple((cam_pos_w @ _R_CONV).tolist())
    fwd_gs    = cam_fwd_w @ _R_CONV
    up_gs     = tuple((cam_up_w  @ _R_CONV).tolist())
    lookat_gs = tuple((cam_pos_w @ _R_CONV + fwd_gs).tolist())
    fov       = float(2 * np.degrees(np.arctan2(cy, fy)))

    render_w = int(round(img_h * fx / fy))

    print(f"[INFO] cameras.json[{frame_idx}] -> pos={pos_gs}, lookat={lookat_gs}, "
          f"up={up_gs}, fov={fov:.1f}, res={img_w}x{img_h}, render_w={render_w}")

    return {
        "pos": pos_gs,
        "lookat": lookat_gs,
        "up": up_gs,
        "fwd": fwd_gs,
        "fov": fov,
        "img_w": img_w,
        "img_h": img_h,
        "render_w": render_w,
        "fx": fx,
        "fy": fy,
    }


# ---------- mesh 最低点 ----------
def find_mesh_lowest_z(mesh_dir):
    """Scan all ``object_*/mesh.obj`` under *mesh_dir* and return the global
    minimum Z coordinate.  Returns 0.0 if no meshes are found."""
    z_min = float('inf')
    for folder in sorted(os.listdir(mesh_dir)):
        if not folder.startswith("object_"):
            continue
        obj_file = os.path.join(mesh_dir, folder, "mesh.obj")
        if not os.path.isfile(obj_file):
            continue
        with open(obj_file, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    z = float(line.split()[3])
                    if z < z_min:
                        z_min = z
    if z_min == float('inf'):
        z_min = 0.0
    print(f"[INFO] Mesh lowest Z = {z_min:.4f}")
    return z_min


# ---------- 背景渲染函数 ----------
def render_with_clean_shadows(cam, bg, shadow_thresh=0.3, shadow_strength=0.3):
    """
    最干净版本：只保留物体 (ID=2) 和阴影（来自平面 ID=1），平面完全去掉。
    背景替换平面区域，阴影作为透明层叠加回去。

    If the render resolution differs from *bg* (e.g. render_w != img_w due to
    fx/fy mismatch correction), the composited result is resized to match *bg*.
    """

    rgb_arr, _, seg_arr, _ = cam.render(segmentation=True, colorize_seg=False)

    rh, rw = rgb_arr.shape[:2]
    # Composite at render resolution using bg scaled to (rw, rh)
    bg_render = cv2.resize(bg, (rw, rh)).astype(np.float32)

    # Masks
    obj_mask = (seg_arr >= 2).astype(np.float32)[..., None]      # 物体
    plane_mask = (seg_arr == 1)                                  # 平面

    # Brightness
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY) / 255.0

    # 阴影 = 平面区域 & 亮度低于阈值
    shadow_mask = (plane_mask & (gray < shadow_thresh)).astype(np.float32)[..., None]

    # ---------------------------------------------------------
    # 合成图像：背景为底 → 叠加物体 → 再叠加阴影（乘暗）
    # ---------------------------------------------------------

    # 1) 从背景开始
    final = bg_render.copy()

    # 2) 加入物体：直接覆盖
    final = final * (1 - obj_mask) + rgb_arr.astype(np.float32) * obj_mask

    final = final * (1 - shadow_mask * shadow_strength)

    # Resize to target (bg) size if render resolution differs
    target_h, target_w = bg.shape[:2]
    if rw != target_w or rh != target_h:
        final = cv2.resize(final, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return final.astype(np.uint8)

def render_with_objects(cam):
    """
    只保留物体 (ID>=2)，背景和阴影全部去掉，输出带透明通道的 RGBA 图像。
    """

    rgb_arr, _, seg_arr, _ = cam.render(segmentation=True, colorize_seg=False)
    h, w = rgb_arr.shape[:2]

    # 物体 mask
    obj_mask = (seg_arr >= 2).astype(np.uint8) * 255  # shape: (h, w)

    # 合成 RGBA：RGB 来自渲染结果，A 来自物体 mask
    rgba = np.dstack([rgb_arr, obj_mask])  # shape: (h, w, 4)

    return rgba


def render_object_only(cam, bg, obj_id_min=2):
    """
    返回：
      final_rgb: 只合成物体（不叠加阴影）
      obj_mask:  物体mask (H,W,1) float32 {0,1}
    """
    rgb_arr, _, seg_arr, _ = cam.render(segmentation=True, colorize_seg=False, force_render=True)

    h, w = rgb_arr.shape[:2]
    bg_resized = cv2.resize(bg, (w, h)).astype(np.float32)

    obj_mask = (seg_arr >= obj_id_min).astype(np.float32)[..., None]

    final = bg_resized * (1 - obj_mask) + rgb_arr.astype(np.float32) * obj_mask

    return final.astype(np.uint8), obj_mask


####################################################################################
####################################################################################
# ---------- 相机位置优化 ----------
def optimize_camera_3d(
    cam,
    ref_mask_dir,
    cam_pos,
    cam_lookat,
    cam_up,
    cam_fwd,
    render_size,
    output_path=None,
    fwd_coarse_range=(-2.0, 2.0),
    fwd_coarse_steps=81,
    fwd_fine_half=0.06,
    fwd_fine_steps=41,
    lat_coarse_range=(-0.3, 0.3),
    lat_coarse_steps=31,
    lat_fine_half=0.02,
    lat_fine_steps=21,
    n_rounds=2,
    shift_max=20,
):
    """Optimize camera position in 3D (forward + right + up) via coordinate
    descent, then refine with sub-pixel shift via cross-correlation.

    Returns
    -------
    best_pos : tuple
    best_lookat : tuple
    best_iou : float
    best_shift : (int, int) – pixel shift (dx, dy) to apply after rendering
    """
    w, h = render_size

    # ---- Load reference mask ----
    ref_mask = _load_ref_mask(ref_mask_dir)
    if ref_mask is None:
        print("[WARN] No reference mask found, skipping camera optimisation.")
        return cam_pos, cam_lookat, 0.0, (0, 0)

    ref_resized = cv2.resize(ref_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    ref_binary = (ref_resized > 127).astype(np.float32)
    print(f"[INFO] Reference mask loaded, non-zero pixels: {int(ref_binary.sum())}")

    # ---- Build orthonormal axes ----
    fwd_unit = cam_fwd / np.linalg.norm(cam_fwd)
    up_vec = np.array(cam_up, dtype=np.float64)
    right_unit = np.cross(fwd_unit, up_vec)
    right_norm = np.linalg.norm(right_unit)
    if right_norm < 1e-6:
        # Fallback: pick an arbitrary perpendicular
        right_unit = np.array([1, 0, 0], dtype=np.float64)
        right_unit -= right_unit.dot(fwd_unit) * fwd_unit
        right_norm = np.linalg.norm(right_unit)
    right_unit /= right_norm
    up_unit = np.cross(right_unit, fwd_unit)
    up_unit /= np.linalg.norm(up_unit)

    pos_base = np.array(cam_pos, dtype=np.float64)
    lookat_base = np.array(cam_lookat, dtype=np.float64)

    # Current offset along each axis (accumulated)
    offsets = np.zeros(3, dtype=np.float64)  # [fwd, right, up]
    axes = [fwd_unit, right_unit, up_unit]
    axis_names = ["Forward", "Right", "Up"]

    def _apply_offset():
        disp = offsets[0] * axes[0] + offsets[1] * axes[1] + offsets[2] * axes[2]
        return pos_base + disp, lookat_base + disp

    def _eval_iou_at(pos, lookat):
        cam.set_pose(pos=tuple(pos.tolist()), lookat=tuple(lookat.tolist()), up=cam_up)
        _, _, seg, _ = cam.render(segmentation=True, colorize_seg=False)
        rend = (seg >= 2).astype(np.float32)
        if rend.shape[:2] != (h, w):
            rend = cv2.resize(rend, (w, h), interpolation=cv2.INTER_NEAREST)
        inter = np.sum(rend * ref_binary)
        union = np.sum(np.clip(rend + ref_binary, 0, 1))
        return inter / (union + 1e-8)

    # ---- Initial IoU ----
    p, l = _apply_offset()
    iou_init = _eval_iou_at(p, l)
    print(f"[OPT] Initial IoU = {iou_init:.4f}")

    # ---- Coordinate descent ----
    # Define search configs per axis
    search_cfg = [
        (fwd_coarse_range, fwd_coarse_steps, fwd_fine_half, fwd_fine_steps),
        (lat_coarse_range, lat_coarse_steps, lat_fine_half, lat_fine_steps),
        (lat_coarse_range, lat_coarse_steps, lat_fine_half, lat_fine_steps),
    ]

    best_iou = iou_init
    for rnd in range(n_rounds):
        for ax_idx in range(3):
            coarse_range, coarse_steps, fine_half, fine_steps = search_cfg[ax_idx]

            # Coarse search
            best_t = offsets[ax_idx]
            for t in np.linspace(coarse_range[0], coarse_range[1], coarse_steps):
                offsets[ax_idx] = t
                p, l = _apply_offset()
                iou = _eval_iou_at(p, l)
                if iou > best_iou:
                    best_iou = iou
                    best_t = t
            offsets[ax_idx] = best_t

            # Fine search
            for t in np.linspace(best_t - fine_half, best_t + fine_half, fine_steps):
                offsets[ax_idx] = t
                p, l = _apply_offset()
                iou = _eval_iou_at(p, l)
                if iou > best_iou:
                    best_iou = iou
                    best_t = t
            offsets[ax_idx] = best_t
            print(f"[OPT] Round {rnd+1} {axis_names[ax_idx]}: t={best_t:.4f}, IoU={best_iou:.4f}")

    best_pos_arr, best_lookat_arr = _apply_offset()
    best_pos = tuple(best_pos_arr.tolist())
    best_lookat = tuple(best_lookat_arr.tolist())
    cam.set_pose(pos=best_pos, lookat=best_lookat, up=cam_up)

    print(f"[OPT] 3D search done: IoU={best_iou:.4f}")

    # ---- Post-render pixel shift via cross-correlation ----
    dx, dy = 0, 0
    if shift_max > 0:
        _, _, seg_opt, _ = cam.render(segmentation=True, colorize_seg=False)
        rend_binary = (seg_opt >= 2).astype(np.float32)
        if rend_binary.shape[:2] != (h, w):
            rend_binary = cv2.resize(rend_binary, (w, h), interpolation=cv2.INTER_NEAREST)

        # Cross-correlation via FFT to find best shift
        corr = fftconvolve(ref_binary, rend_binary[::-1, ::-1], mode='full')
        cy_corr, cx_corr = np.array(corr.shape) // 2  # zero-shift location
        # Restrict search to ±shift_max
        y0 = max(0, cy_corr - shift_max)
        y1 = min(corr.shape[0], cy_corr + shift_max + 1)
        x0 = max(0, cx_corr - shift_max)
        x1 = min(corr.shape[1], cx_corr + shift_max + 1)
        roi = corr[y0:y1, x0:x1]
        peak_y, peak_x = np.unravel_index(np.argmax(roi), roi.shape)
        dy = (peak_y + y0) - cy_corr
        dx = (peak_x + x0) - cx_corr

        # Verify with true IoU
        if dx != 0 or dy != 0:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(rend_binary, M, (w, h))
            inter_s = np.sum(shifted * ref_binary)
            union_s = np.sum(np.clip(shifted + ref_binary, 0, 1))
            iou_shifted = inter_s / (union_s + 1e-8)
            if iou_shifted > best_iou:
                best_iou = iou_shifted
                print(f"[OPT] Pixel shift: dx={dx}, dy={dy}, IoU={best_iou:.4f}")
            else:
                dx, dy = 0, 0
                print(f"[OPT] Pixel shift rejected (no IoU gain)")
        else:
            print(f"[OPT] Pixel shift: (0, 0), no shift needed")

    # ---- Save comparison image ----
    if output_path is not None:
        _, _, seg_opt, _ = cam.render(segmentation=True, colorize_seg=False)
        rend_binary = (seg_opt >= 2).astype(np.float32)
        if rend_binary.shape[:2] != (h, w):
            rend_binary = cv2.resize(rend_binary, (w, h), interpolation=cv2.INTER_NEAREST)
        if dx != 0 or dy != 0:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            rend_binary = cv2.warpAffine(rend_binary, M, (w, h))
        compare = np.zeros((h, w, 3), dtype=np.uint8)
        compare[:, :, 1] = (ref_binary * 255).astype(np.uint8)       # green = reference
        compare[:, :, 2] = (rend_binary * 255).astype(np.uint8)      # red   = render
        # yellow = overlap
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, compare)
        print(f"[OPT] Comparison saved to {output_path}")

    print(f"[OPT] Final: pos={best_pos}, shift=({dx},{dy}), IoU={best_iou:.4f}")
    return best_pos, best_lookat, best_iou, (dx, dy)


def _load_ref_mask(ref_mask_dir):
    """Load reference binary mask from rgba_masks directory."""
    ref_mask = None
    combined_path = os.path.join(ref_mask_dir, "combined_binary.png")
    if os.path.isfile(combined_path):
        ref_mask = cv2.imread(combined_path, cv2.IMREAD_GRAYSCALE)

    if ref_mask is None and os.path.isdir(ref_mask_dir):
        first = True
        for mf in sorted(os.listdir(ref_mask_dir)):
            if not mf.endswith('.png') or 'combined' in mf:
                continue
            m = cv2.imread(os.path.join(ref_mask_dir, mf), cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            alpha = m[:, :, 3] if (m.ndim == 3 and m.shape[2] == 4) else m
            layer = (alpha > 127).astype(np.uint8) * 255
            if first:
                combined = layer
                first = False
            else:
                combined = np.maximum(combined, layer)
        if not first:
            ref_mask = combined

    if ref_mask is not None and ref_mask.max() == 0:
        return None
    return ref_mask