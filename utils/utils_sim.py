import os
import re
import math
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ---------- 背景渲染函数 ----------
def render_with_clean_shadows(cam, bg, shadow_thresh=0.3, shadow_strength=0.3):
    """
    最干净版本：只保留物体 (ID=2) 和阴影（来自平面 ID=1），平面完全去掉。
    背景替换平面区域，阴影作为透明层叠加回去。
    """

    rgb_arr, _, seg_arr, _ = cam.render(segmentation=True,colorize_seg=False)
    h, w = rgb_arr.shape[:2]
    bg_resized = cv2.resize(bg, (w, h)).astype(np.float32)

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
    final = bg_resized.copy()

    # 2) 加入物体：直接覆盖
    final = final * (1 - obj_mask) + rgb_arr.astype(np.float32) * obj_mask

    final = final * (1 - shadow_mask * shadow_strength)

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


def crop_by_camera_offset(
    img,
    cpos,
    crop_h,
    crop_w,
):

    H, W = img.shape[:2]

    cam_x = cpos[0]
    cam_y = cpos[1]
    cam_z = cpos[2]        


    x_center = W // 2
    max_x = (W - crop_w)//2
    x1 = int(x_center - crop_w / 2) + int(cam_x * max_x)
    x2 = x1 + crop_w


    y_center = H // 2
    y1 = int(y_center - crop_h / 2)
    y2 = int(y_center + crop_h / 2)


    return img[y1:y2, x1:x2]


def compute_mesh_pos_from_obj(obj_path):
    # 读取 mesh
    mesh = trimesh.load(obj_path, force='mesh')

    verts = mesh.vertices.copy()
    
    center = verts.mean(axis=0)
    min_z = verts[:, 2].max()

    pos = np.array([
        -center[0],
        -center[1],
        min_z
    ])

    return pos, center, min_z



def mesh_alignment(mesh_path):
    """
    从 mesh 计算：
    1. 几何中心 center
    2. 主轴方向 axis（PCA）
    3. 将该主轴对齐到 z-up 的欧拉角（X-Y-Z 顺序）
    Returns
    -------
    center : (3,)
        mesh 几何中心
    axis : (3,)
        主轴方向（单位向量，方向已统一）
    euler : tuple
        (rx, ry, rz) in degrees
    """

    # ---------- 1. 加载 mesh ----------
    mesh = trimesh.load(mesh_path, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        )

    vertices = mesh.vertices  # (N, 3)

    # ---------- 2. 计算几何中心 ----------
    center = vertices.mean(axis=0)
    centered = vertices - center

    # mesh.vertices = centered
    # tmp_dir = tempfile.gettempdir()          # 系统 tmp 目录
    # save_path = os.path.join(tmp_dir, "normalized_mesh.obj")
    # mesh.export(save_path)

    # ---------- 3. PCA / SVD 求主轴 ----------
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    axis = axis / np.linalg.norm(axis)

    # ---------- 4. 统一主轴方向（指向 -Z） ----------
    world_head_dir = np.array([0.0, 0.0, 1.0])
    if np.dot(axis, world_head_dir) > 0:
        axis = -axis

    # ---------- 5. 计算对齐到 Z 的欧拉角 ----------
    vx, vy, vz = axis

    # 绕 X：消除 Y
    rx = np.arctan2(vy, vz)

    # 绕 Y：消除 X
    ry = -np.arctan2(vx, np.sqrt(vy**2 + vz**2))
    rz = 0.0
    euler = (
        np.degrees(rx),
        np.degrees(ry),
        np.degrees(rz)
    )

     # ---------- 6. 应用 euler 到 mesh，并计算最低点 ----------
    # trimesh.transformations.euler_matrix 默认是静态轴 'sxyz'（对应 X-Y-Z）
    R = trimesh.transformations.euler_matrix(
        np.radians(euler[0]),
        np.radians(euler[1]),
        np.radians(euler[2]),
        axes='sxyz'
    )

    # 把 centered 顶点旋转（齐次坐标）
    v_h = np.hstack([centered, np.ones((centered.shape[0], 1))])   # (N,4)
    rotated = (R @ v_h.T).T[:, :3]                                 # (N,3)

    # 最低点（z 最小）
    idx = np.argmin(rotated[:, 2])
    lowest_point = rotated[idx]

    # 1) 最低点到原点的欧氏距离
    lowest_to_origin_dist = float(np.linalg.norm(lowest_point))

    # 2) 只看竖直方向距离（“离地高度”）
    lowest_vertical_dist = float(abs(lowest_point[2]))


    return center, euler, lowest_vertical_dist ,axis



def pca_axis(mesh_path):

    # ---------- 1. 加载 mesh ----------
    mesh = trimesh.load(mesh_path, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        )

    vertices = mesh.vertices  # (N, 3)

    # ---------- 2. 计算几何中心 ----------
    center = vertices.mean(axis=0)
    centered = vertices - center


    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    axis = axis / np.linalg.norm(axis)

    # ---------- 4. 统一主轴方向（指向 -Z） ----------
    world_head_dir = np.array([0.0, 0.0, 1.0])
    if np.dot(axis, world_head_dir) > 0:
        axis = -axis

    return axis,center



def axiz_to_e(rotation_axis, target_axis):
    # 计算旋转轴 (垂直于这两个向量的轴)
    axis_of_rotation = np.cross(rotation_axis, target_axis)
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)  # 归一化

    # 计算旋转角度
    cos_angle = np.dot(rotation_axis, target_axis)
    angle = np.arccos(cos_angle)

    # 使用旋转轴和角度构建旋转矩阵
    rotation = R.from_rotvec(axis_of_rotation * angle)

    # 获取欧拉角 (使用'xyz'表示顺序，可以根据需要选择其他顺序)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    
    return euler_angles




def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    返回旋转矩阵 R，使得 R @ a ≈ b
    a, b: shape (3,)
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    # 已经对齐
    if np.isclose(c, 1.0):
        return np.eye(3)

    # 反向 (180°)
    if np.isclose(c, -1.0):
        # 任意找一个与 a 不共线的轴
        axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(a, axis):
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        return -np.eye(3) + 2 * np.outer(v, v)

    s = np.linalg.norm(v)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R


def norm_mesh(mesh_path_1: str,
              mean_center: np.ndarray,
              mean_axiz: np.ndarray,
              target_axis: np.ndarray = np.array([0.0, 0.0, 1.0])):

    mean_center = np.asarray(mean_center, dtype=np.float64)
    mean_axiz = np.asarray(mean_axiz, dtype=np.float64)
    target_axis = np.asarray(target_axis, dtype=np.float64)

    R = rotation_matrix_from_vectors(mean_axiz, target_axis)

    
    out_path = mesh_path_1.replace(".obj", "_norm.obj")

    z_min = np.inf

    with open(mesh_path_1, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if line.startswith("v "):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                v = np.array([x, y, z], dtype=np.float64)

                v = v - mean_center
                v = R @ v

                if v[2] < z_min:
                    z_min = v[2]

                fout.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            else:
                fout.write(line)

    lift_to_ground = max(0.0, -z_min)

    base_name = os.path.splitext(os.path.basename(mesh_path_1))[0]
    dir_name = os.path.dirname(mesh_path_1)
    lift_path = os.path.join(dir_name, f"{base_name}_lift.txt")

    with open(lift_path, "w", encoding="utf-8") as f:
        f.write(f"{lift_to_ground:.6f}\n")


def fit_scene_plane_normal_from_objects(
    mesh_path: str,
    points_per_mesh: int = 5000,          # 每个物体采样多少点
    max_meshes: int | None = None,        # 最多用多少个mesh（None=全用）
    ransac_dist_thresh: float = 0.01,     # RANSAC 内点距离阈值（单位同mesh坐标单位）
    ransac_n: int = 3,
    ransac_iters: int = 2000,
    min_total_points: int = 5000,         # 点太少就不拟合
    up_hint: Optional[np.ndarray] = None, # 统一法线朝向，如 np.array([0,1,0])
    low_percentile: float = 5.0,          # 方案A：取最低多少百分位的点（建议 2~10）
    low_percentile_fallback: float = 10.0,# 低层点太少时的放宽百分位
    horizontal_cos_thresh: float = 0.9    # 方案B：水平性阈值，|n·up| >= 0.85~0.9
) -> Tuple[np.ndarray, Tuple[float, float, float, float], np.ndarray]:
    """
    返回:
      normal: (3,) 单位法线（方向会按 up_hint 统一到同一半球）
      plane:  (a,b,c,d) 平面方程 ax+by+cz+d=0（已归一化）
      centroid_all: (3,) 全部采样点的质心
    """

    # 默认 up 方向是 +Y
    if up_hint is None:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        up_hint = np.asarray(up_hint, dtype=np.float64)

    up_norm = np.linalg.norm(up_hint)
    if up_norm < 1e-12:
        raise ValueError("up_hint is degenerate.")
    up_hint = up_hint / up_norm

    all_points = []
    used = 0

    for folder in sorted(os.listdir(mesh_path)):
        if not folder.startswith("object_"):
            continue

        if re.search(r"\d+$", folder) is None:
            continue

        mesh_obj = os.path.join(mesh_path, folder, "mesh.obj")
        if not os.path.isfile(mesh_obj):
            continue

        mesh = o3d.io.read_triangle_mesh(mesh_obj)
        if mesh.is_empty() or (not mesh.has_triangles()):
            continue

        pcd = mesh.sample_points_uniformly(number_of_points=points_per_mesh)
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            continue

        all_points.append(pts)
        used += 1
        if max_meshes is not None and used >= max_meshes:
            break

    if len(all_points) == 0:
        raise RuntimeError("No valid meshes found to fit a plane.")

    pts = np.concatenate(all_points, axis=0)
    if pts.shape[0] < min_total_points:
        raise RuntimeError(
            f"Not enough points to fit plane: {pts.shape[0]} < {min_total_points}"
        )

    centroid_all = pts.mean(axis=0)

    # 2) 方案A：只用最低层点拟合（按 up_hint 对应的“高度”投影来取最低层）
    #    高度标量 = 点在 up 方向上的投影
    heights = pts @ up_hint  # (N,)
    y_thr = np.percentile(heights, low_percentile)
    low_pts = pts[heights <= y_thr]

    # 低层点不够就放宽
    if low_pts.shape[0] < max(500, min_total_points // 5):
        y_thr2 = np.percentile(heights, low_percentile_fallback)
        low_pts = pts[heights <= y_thr2]

    if low_pts.shape[0] < max(200, ransac_n * 10):
        raise RuntimeError(
            f"Too few low-layer points for RANSAC: {low_pts.shape[0]}. "
            f"Try increasing points_per_mesh or low_percentile(_fallback)."
        )

    low_pcd = o3d.geometry.PointCloud()
    low_pcd.points = o3d.utility.Vector3dVector(low_pts)

    plane_model, inliers = low_pcd.segment_plane(
        distance_threshold=ransac_dist_thresh,
        ransac_n=ransac_n,
        num_iterations=ransac_iters
    )

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-12:
        raise RuntimeError("Degenerate plane normal.")
    normal /= n_norm
    d /= n_norm

    # 3) 统一法线方向：让 normal 与 up_hint 同向（点积为正）
    if normal.dot(up_hint) < 0:
        normal = -normal
        d = -d

    # 4) 方案B：水平性检查（地面法线应与 up_hint 接近）
    #    如果 normal 与 up_hint 夹角太大，说明拟合到的不是地面
    cos_val = abs(normal.dot(up_hint))
    if cos_val < horizontal_cos_thresh:
        raise RuntimeError(
            f"Fitted plane is not horizontal enough: |n·up|={cos_val:.3f} "
            f"< {horizontal_cos_thresh}. "
            f"Try increasing low_percentile, lowering ransac_dist_thresh, or removing humans."
        )

    print(
        f"[GroundPlane] a={normal[0]:.6f}, b={normal[1]:.6f}, c={normal[2]:.6f}, d={d:.6f} | "
        f"|n·up|={cos_val:.3f} | inliers={len(inliers)}/{low_pts.shape[0]}"
    )

    return normal, (float(normal[0]), float(normal[1]), float(normal[2]), float(d)), centroid_all



def resolve_penetration(mesh_path: str, padding: float = 0.01, max_delta: float = 0.05):
    """
    1. 检测所有 _norm.obj 之间的穿模。
    2. 计算平移向量，限制幅度以保持相对位置。
    3. 通过直接修改 v 行来保存，确保材质 (mtl) 不丢失。
    """
    # --- 1. 使用 Open3D 获取每个物体的包围盒 ---
    objects_info = []
    folder_list = sorted([f for f in os.listdir(mesh_path) if f.startswith("object_")])
    
    for folder in folder_list:
        obj_path = os.path.join(mesh_path, folder, "mesh_norm.obj")
        if not os.path.isfile(obj_path): continue
        
        # 仅加载点云或简易 mesh 用于计算包围盒
        mesh = o3d.io.read_triangle_mesh(obj_path)
        if mesh.is_empty(): continue
        
        bbox = mesh.get_axis_aligned_bounding_box()
        objects_info.append({
            "path": obj_path,
            "center": bbox.get_center(),
            "extent": bbox.get_extent() + padding,
            "delta": np.zeros(3)
        })

    if len(objects_info) < 2:
        return

    # --- 2. 局部斥力算法：两两检测并推开 ---
    # 迭代 2 次以处理链式碰撞
    for _ in range(2):
        for i in range(len(objects_info)):
            for j in range(i + 1, len(objects_info)):
                a, b = objects_info[i], objects_info[j]
                
                diff = a["center"] - b["center"]
                dist = np.abs(diff)
                min_dist = (a["extent"] + b["extent"]) / 2.0
                
                overlap = min_dist - dist
                if np.all(overlap > 0):
                    # 寻找重叠最少的轴（最小干预原则）
                    axis = np.argmin(overlap)
                    push_val = overlap[axis] * 0.5
                    
                    # 构造推力
                    push_vec = np.zeros(3)
                    push_vec[axis] = push_val if diff[axis] > 0 else -push_val
                    
                    # 累加位移
                    a["delta"] += push_vec
                    b["delta"] -= push_vec
                    # 同步更新中心点用于下一轮计算
                    a["center"] += push_vec
                    b["center"] -= push_vec

    # --- 3. 修改文件：直接读写文本，保留材质行 ---
    for obj in objects_info:
        # 限制位移幅度
        move_len = np.linalg.norm(obj["delta"])
        if move_len < 1e-6: continue # 无位移则跳过
        
        actual_move = obj["delta"]
        if move_len > max_delta:
            actual_move = (actual_move / move_len) * max_delta
            
        print(f"Moving {os.path.basename(obj['path'])} by {np.linalg.norm(actual_move):.4f}m to fix overlap")

        tmp_path = obj["path"] + ".tmp"
        with open(obj["path"], "r", encoding="utf-8", errors="ignore") as fin, \
             open(tmp_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.startswith("v "):
                    parts = line.split()
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    # 应用位移
                    v_new = v + actual_move
                    fout.write(f"v {v_new[0]:.6f} {v_new[1]:.6f} {v_new[2]:.6f}\n")
                else:
                    fout.write(line)
        
        os.remove(obj["path"])
        os.rename(tmp_path, obj["path"])

    # print("穿模修复完成，材质已保留。")


def normalize_scene_to_ground(
    mesh_path: str,
    normal,
    centroid_all,
):
   
    target_up = np.array([0.0, 0.0, 1.0])
    # 2. 计算旋转矩阵
    R = rotation_matrix_from_vectors(normal, target_up)

    # 3. 第一次遍历：计算全局 z_min (旋转后的)
    global_z_min = np.inf
    valid_tasks = []

    for folder in sorted(os.listdir(mesh_path)):
        if not folder.startswith("object_"): continue
        obj_path = os.path.join(mesh_path, folder, "mesh.obj")
        if not os.path.isfile(obj_path): continue

        # 仅读取顶点来寻找 z_min
        with open(obj_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    # 变换坐标: v_new = R @ (v - centroid)
                    v_new = (v - centroid_all) @ R.T
                    if v_new[2] < global_z_min:
                        global_z_min = v_new[2]
        
        valid_tasks.append((obj_path, obj_path.replace(".obj", "_norm.obj")))

    for src_path, dst_path in valid_tasks:
        with open(src_path, "r", encoding="utf-8", errors="ignore") as fin, \
             open(dst_path, "w", encoding="utf-8") as fout:
            
            for line in fin:
                if line.startswith("v "):
                    parts = line.split()
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    # 应用 旋转 -> 移至原点 -> 抬升至 z=0
                    v_new = (v - centroid_all) @ R.T
                    v_new[2] -= global_z_min
                    
                    fout.write(f"v {v_new[0]:.6f} {v_new[1]:.6f} {v_new[2]:.6f}\n")
                else:
                    fout.write(line)

    info_path = os.path.join(mesh_path, "scene_norm_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"{global_z_min:.6f}\n")





def compute_zmmmm(mesh_file):
    """
    返回 zmmmm，使 mesh 最低点刚好贴到 z=0
    """
    mesh = trimesh.load(mesh_file, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_file}")

    V = np.asarray(mesh.vertices, dtype=np.float64)

    min_z = V[:, 2].min()
    zmmmm = -min_z
    return zmmmm




def transform_camera_with_mesh(
    pos1,
    lookat1,
    mean_center,
    mean_axis,
    target_axis=np.array([0.0, 0.0, 1.0])
):
    pos1 = np.asarray(pos1, dtype=np.float64)
    lookat1 = np.asarray(lookat1, dtype=np.float64)
    mean_center = np.asarray(mean_center, dtype=np.float64)
    mean_axis = np.asarray(mean_axis, dtype=np.float64)
    target_axis = np.asarray(target_axis, dtype=np.float64)


    # mesh 使用的旋转
    R = rotation_matrix_from_vectors(mean_axis, target_axis)

    pos2 = R @ (pos1 - mean_center)

    lookat2 = R @ lookat1
    n = np.linalg.norm(lookat2)
    if n > 1e-12:
        lookat2 /= n

    return pos2, lookat2



def move_pos_along_view(pos2, lookat2, move_dist):
    """
    pos2: [x, y, z]
    lookat2: [x, y, z]
    move_dist: 控制移动距离（正数向远离 lookat 方向，负数相反）
    """
    # 方向向量 pos2 - lookat2
    dx = pos2[0] - lookat2[0]
    dy = pos2[1] - lookat2[1]
    dz = pos2[2] - lookat2[2]

    # 向量长度
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0:
        return pos2  # 防止除 0

    # 单位方向向量
    dx /= length
    dy /= length
    dz /= length

    # 沿方向移动
    pos2[0] += dx * move_dist
    pos2[1] += dy * move_dist
    pos2[2] += dz * move_dist

    return pos2




