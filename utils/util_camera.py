import os
import cv2
import numpy as np
from utils.utils_sim import render_object_only
from scipy.optimize import minimize

def load_target_object_mask(image_mask_path, target_img, thresh=127):
    """
    mask: 白色部分为物体
    return: (H,W,1) float32, 物体=1, 背景=0
    """
    m = cv2.imread(image_mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(image_mask_path)

    if m.ndim == 3:
        if m.shape[2] == 4:
            m = m[:, :, 3]
        else:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    H, W = target_img.shape[:2]
    if m.shape[0] != H or m.shape[1] != W:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    # 白色(>thresh) => 物体(1)
    m = (m > thresh).astype(np.float32)
    return m[:, :, None]



def random_search(obj_fn, x0, bounds, n=60, seed=0):
    rng = np.random.default_rng(seed)
    
    best_x = np.array(x0, dtype=np.float64)
    best_f = obj_fn(best_x)

    lows = np.array([b[0] for b in bounds], dtype=np.float64)
    highs = np.array([b[1] for b in bounds], dtype=np.float64)

    for _ in range(n):
        x = rng.uniform(lows, highs)
        f = obj_fn(x)
        if f < best_f:
            best_x, best_f = x, f
    return best_x, best_f



def refine(obj_fn, x_init, bounds):
    res = minimize(
        obj_fn,
        x0=np.array(x_init, dtype=np.float64),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 80, "disp": True}
    )
    return res.x, res.fun


def _to_float01(img_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB/BGR -> float32 [0,1]"""
    img = np.asarray(img_u8)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def _ensure_hw1(mask: np.ndarray) -> np.ndarray:
    """Ensure (H,W,1)."""
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[..., None]
    elif mask.ndim == 3 and mask.shape[2] != 1:
        mask = mask[..., :1]
    return mask


def normalize_mask01(mask: np.ndarray) -> np.ndarray:
    """
    Normalize mask to float32 [0,1] with shape (H,W,1).
    Accepts: (H,W), (H,W,1), (H,W,3), uint8 0/255, float 0/1, float 0/255.
    """
    mask = _ensure_hw1(mask)

    if mask.dtype == np.uint8:
        m = mask.astype(np.float32) / 255.0
    else:
        m = mask.astype(np.float32)
        if m.size > 0 and (np.nanmax(m) > 1.0 + 1e-3):
            m = m / 255.0

    return np.clip(m, 0.0, 1.0)


def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure uint8 RGB (H,W,3)."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_to_match_hw(img: np.ndarray, H: int, W: int, interp=cv2.INTER_AREA) -> np.ndarray:
    if img.shape[0] == H and img.shape[1] == W:
        return img
    return cv2.resize(img, (W, H), interpolation=interp)


def _mask_norm(m_hw1: np.ndarray, channels: int = 3, eps: float = 1e-6) -> float:
    """Normalization factor by effective pixels * channels (supports soft mask)."""
    m_hw1 = normalize_mask01(m_hw1)
    return float(np.sum(m_hw1) * channels + eps)


# -------------------------
# pixel loss under mask
# -------------------------
def image_mask_loss(a_f: np.ndarray, b_f: np.ndarray, m_hw1: np.ndarray) -> float:
    """
    a_f,b_f: (H,W,3) float [0,1]
    m_hw1:   (H,W,1) float [0,1]
    return: sum_{pixels,channels} ((a-b)*m)^2
    """
    m_hw1 = normalize_mask01(m_hw1)
    d = (a_f - b_f) * m_hw1
    return float(np.sum(d * d))


# -------------------------
# 1) object region loss
# -------------------------
def loss_object_region(
    final_f: np.ndarray,
    target_f: np.ndarray,
    tgt_mask_hw1: np.ndarray,
    image_mask_loss_fn=image_mask_loss,
    eps: float = 1e-6,
) -> float:
    tgt_mask_hw1 = normalize_mask01(tgt_mask_hw1)
    n_obj = _mask_norm(tgt_mask_hw1, channels=3, eps=eps)
    return float(image_mask_loss_fn(final_f, target_f, tgt_mask_hw1) / n_obj)


# -------------------------
# 2) background region loss
# -------------------------
def loss_background_region(
    final_f: np.ndarray,
    target_f: np.ndarray,
    tgt_bg_hw1: np.ndarray,
    image_mask_loss_fn=image_mask_loss,
    eps: float = 1e-6,
) -> float:
    tgt_bg_hw1 = normalize_mask01(tgt_bg_hw1)
    n_bg = _mask_norm(tgt_bg_hw1, channels=3, eps=eps)
    return float(image_mask_loss_fn(final_f, target_f, tgt_bg_hw1) / n_bg)


# -------------------------
# 3) mask alignment loss (Dice)
# -------------------------
def loss_mask_alignment(
    render_obj_mask: np.ndarray,
    tgt_mask_hw1: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Dice loss for mask alignment; robust for small foregrounds.
    render_obj_mask: (H,W) or (H,W,1) uint8{0,255} or float{0,1}/float{0,255}
    tgt_mask_hw1:    (H,W,1) float{0,1} (or any, will be normalized)
    """
    obj_m = normalize_mask01(render_obj_mask)   # (H,W,1)
    tgt_m = normalize_mask01(tgt_mask_hw1)      # (H,W,1)

    inter = float(np.sum(obj_m * tgt_m))
    denom = float(np.sum(obj_m) + np.sum(tgt_m) + eps)
    dice = (2.0 * inter + eps) / denom
    return float(1.0 - dice)



# -------------------------
# main objective builder
# -------------------------
def make_objective(
    cam,
    bg,
    target_img,
    lookat0,
    image_mask_path=None,
    w_obj=1.0,
    w_bg=0.2,
    w_mask=1.0,
    save_dir="opt_debug",
    save_every=5,
):
    """
    cam: has set_pose(pos=[x,y,z], lookat=lookat0)
    bg: background object used by render_object_only
    target_img: uint8 RGB (recommended). If not, will be clamped/cast for debug.
    lookat0: look-at point
    image_mask_path: optional path for target object mask
    render_object_only(cam, bg) must return:
        final: uint8 RGB (H,W,3)
        obj_mask: (H,W) or (H,W,1), uint8 or float
    """
    # os.makedirs(save_dir, exist_ok=True)
    
    counter = {"i": 0}
    trace = []

    # target image
    target_img_u8 = ensure_rgb_uint8(target_img)
    target_f = _to_float01(target_img_u8)
    Ht, Wt = target_img_u8.shape[:2]

    # target mask
    if image_mask_path is not None:
        # user-provided function (must exist in your project)
        raw_mask = load_target_object_mask(image_mask_path, target_img_u8)
        tgt_mask = normalize_mask01(raw_mask)
    else:
        tgt_mask = np.ones((Ht, Wt, 1), np.float32)

    tgt_bg = 1.0 - tgt_mask

    # precompute target mask visualization (constant)
    tgt_vis = (tgt_mask[..., 0] * 255.0).clip(0, 255).astype(np.uint8)
    tgt_vis = cv2.cvtColor(tgt_vis, cv2.COLOR_GRAY2RGB)

    def objective(v):
        counter["i"] += 1
        i = counter["i"]

        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.size == 4:
            x, y, z, fov = map(float, v[:4])
            cam.set_pose(pos=[x, y, z], lookat=lookat0)
            cam._fov = fov
        elif v.size == 3:
            x, y, z = map(float, v[:3])
            cam.set_pose(pos=[x, y, z], lookat=lookat0)
        else:
            raise ValueError(f"Expected v with 3 or 4 elements, got {v.shape}")

        # user-provided renderer (must exist in your project)
        final, obj_mask = render_object_only(cam, bg)

        # ensure size match for loss computation
        final_u8 = ensure_rgb_uint8(final)
        if final_u8.shape[:2] != (Ht, Wt):
            final_u8 = resize_to_match_hw(final_u8, Ht, Wt, interp=cv2.INTER_AREA)

        final_f = _to_float01(final_u8)

        # losses
        l_obj = loss_object_region(final_f, target_f, tgt_mask, image_mask_loss)
        l_bg = loss_background_region(final_f, target_f, tgt_bg, image_mask_loss) if w_bg != 0 else 0.0
        l_msk = loss_mask_alignment(obj_mask, tgt_mask) if w_mask != 0 else 0.0

        loss = float(w_obj * l_obj + w_bg * l_bg + w_mask * l_msk)

        # --- debug visualization ---
        # if save_every and (i % save_every == 0):

        #     trace.append({
        #         "i": i,
        #         "loss": loss,
        #         "v": [x, y, z],
        #         "l_obj": float(l_obj),
        #         "l_bg": float(l_bg),
        #         "l_msk": float(l_msk),
        #     })

    
        #     # obj_mask visualization (resized to match target for hstack)
        #     om = normalize_mask01(obj_mask)
        #     if om.shape[:2] != (Ht, Wt):
        #         om = resize_to_match_hw(om, Ht, Wt, interp=cv2.INTER_NEAREST)
        #     om_vis = (om[..., 0] * 255.0).clip(0, 255).astype(np.uint8)
        #     om_vis = cv2.cvtColor(om_vis, cv2.COLOR_GRAY2RGB)

        #     side_by_side = np.hstack([target_img_u8, final_u8, tgt_vis, om_vis])

        #     vis_path = (
        #         f"{save_dir}/iter_{i:05d}_loss_{loss:.6f}"
        #         f"_obj_{float(l_obj):.6f}_bg_{float(l_bg):.6f}_msk_{float(l_msk):.6f}.png"
        #     )
        #     # side_by_side is RGB; OpenCV wants BGR
        #     cv2.imwrite(vis_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

        return loss
    
    objective.trace = trace
    objective.counter = counter

    return objective
