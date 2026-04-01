import os
import numpy as np
from utils.utils_sim import (
    fit_scene_plane_normal_from_objects, 
    transform_camera_with_mesh, 
)
import time

try:
    from camera_alignment import align_camera
    _NEW_ALIGN_AVAILABLE = True
except ImportError:
    _NEW_ALIGN_AVAILABLE = False
    print("[WARN] camera_alignment module not found, new align mode unavailable.")



def _get_camera_legacy(mesh_path, sim_cfg, z_min_global):
    """
    旧方法：拟合地面法向量 → transform_camera_with_mesh → 启发式位姿
    
    Returns:
        pos2 (list[float]): 相机位置 [x, y, z]
        lookat2 (list[float]): 注视点 [x, y, z]
        fov (float): 视场角（固定60度）
        method (str): 使用的方法名称
    """
    dist_t = sim_cfg.get('dist_thresh', 0.01)
    cos_t  = sim_cfg.get('cos_thresh', 0.8)
    low_p  = sim_cfg.get('low_p', 5)

    normal, plane, mean_center = None, None, None
    for i in range(sim_cfg.get('max_retries', 8)):
        try:
            normal, plane, mean_center = fit_scene_plane_normal_from_objects(
                mesh_path,
                points_per_mesh=5000,
                ransac_dist_thresh=dist_t,
                up_hint=np.array([0, 0, 1]),
                low_percentile=low_p,
                low_percentile_fallback=low_p + 5,
                horizontal_cos_thresh=cos_t,
            )
            break
        except Exception as e:
            if i == sim_cfg.get('max_retries', 8) - 1:
                raise e
            low_p  += 5
            dist_t += 0.005
            cos_t   = max(0.5, cos_t - 0.05)
            time.sleep(0.1)

    pos2, lookat2 = transform_camera_with_mesh(
        (0, 0, 0), (0, -1, 0), mean_center, normal
    )

    # 应用 z_min 补偿
    pos2[2]    -= z_min_global
    lookat2[2] -= z_min_global

    return pos2, lookat2, 60.0, "legacy", mean_center, normal


def _get_camera_new(image_path, mask_dir, mesh_path, sim_cfg, z_min_global, device="cuda"):
    """
    新方法：3层级对齐（DA3先验 → PnP → 启发式）
    
    Returns:
        pos2 (list[float]): 相机位置 [x, y, z]
        lookat2 (list[float]): 注视点 [x, y, z]
        fov (float): 视场角
        method (str): 实际使用的子方法
    """
    if not _NEW_ALIGN_AVAILABLE:
        raise RuntimeError(
            "camera_alignment module not available. "
            "Please check utils/camera_alignment.py exists."
        )

    transform_path = os.path.join(mesh_path, "scene_transform.npz")

    result = align_camera(
        image_path=image_path,
        mask_dir=mask_dir,
        mesh_dir=mesh_path,
        scene_transform_path=transform_path if os.path.isfile(transform_path) else None,
        image_size=(880, 880),
        device=device,
    )

    pos2    = list(result['position'])
    lookat2 = list(result['lookat'])
    fov     = float(result['fov'])
    method  = f"new:{result['method']}"  # e.g. "new:pnp", "new:da3", "new:heuristic"

    # 应用 z_min 补偿（新方法坐标系与旧方法相同，都需要补偿）
    pos2[2]    -= z_min_global
    lookat2[2] -= z_min_global

    print(f"[INFO] New camera align → method={result['method']}, "
          f"pos={pos2}, lookat={lookat2}, fov={fov:.1f}°")

    return pos2, lookat2, fov, method

