import argparse
import genesis as gs
import cv2
import os
import re
import time
import numpy as np
import yaml
from PIL import Image
from utils.utils_sim import render_with_clean_shadows,render_with_objects
from utils.utils_sim import (
    fit_scene_plane_normal_from_objects,
    normalize_scene_to_ground,
    resolve_penetration,
    move_pos_along_view
)

from utils.camera_es import _get_camera_legacy,_get_camera_new
from utils.physics import DEFAULT_CONFIG, SURFACE_DEFAULTS, get_material, get_force_field


def run_scene(
    root_dir,
    scene_name,
    config=None,
    output_dir="demo/demo_plus",
    move=0,
    camera_align="legacy",
    device="cuda",
):
    BASE_PATH = f"{root_dir}/{scene_name}"

    if config is not None:
        config_path = config
    else:
        config_path = os.path.join(BASE_PATH, "config.yaml")

    mesh_path        = os.path.join(BASE_PATH, "sam3d")
    image_path       = os.path.join(BASE_PATH, f"{scene_name}.png")
    image_mask_path  = os.path.join(BASE_PATH, "rgba_masks", "combined_binary.png")
    mask_dir         = os.path.join(BASE_PATH, "rgba_masks")

    # ---- Load config ----
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        print(f"[INFO] Config loaded from {config_path}")
    else:
        cfg = DEFAULT_CONFIG
        print(f"[WARN] Config not found, using default settings.")

    sim_cfg = cfg.get('simulation', DEFAULT_CONFIG['simulation'])
    obj_cfg = cfg.get('objects', {})

    gs.init(eps=1e-12, backend=gs.cuda)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10, gravity=(0, 0, -9.8)),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
            particle_size=0.01,
        ),
        vis_options=gs.options.VisOptions(
            segmentation_level="entity",
            background_color=(0, 0, 0),
            ambient_light=(1.0, 1.0, 1.0),
            lights=[{
                "type": "directional",
                "dir": (-1, -1, -1),
                "color": (1.0, 1.0, 1.0),
                "intensity": 0.0,
            }]
        ),
        show_viewer=False,
        coupler_options=gs.options.LegacyCouplerOptions(
                rigid_pbd=True,
                rigid_mpm=True,
        )
    )

    # ---- Ground plane ----
    plane_mat = gs.materials.Rigid(friction=0.7)
    if sim_cfg.get('use_plane', True):
        scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0), visualization=False, plane_size=(5,5)), material=plane_mat)
        scene.add_entity(gs.morphs.Mesh(file="utils/plane.obj", pos=(0, 0, 0), euler=(90, 0, 180), scale=20, fixed=True))
    else:
        scene.add_entity(gs.morphs.Mesh(file="utils/plane.obj", pos=(10, 10, 10), euler=(0, 0, 0), scale=1, fixed=True))



    # ---- Load objects (need z_min before camera pose) ----
    entities      = {}
    z_min_global  = 0

    folders = [f for f in sorted(os.listdir(mesh_path)) if f.startswith("object_")]

    for folder in folders:
        match = re.search(r'\d+$', folder)
        if not match:
            continue
        idx = int(match.group())

        mesh_path_3 = os.path.join(mesh_path, folder, "mesh_norm.obj")
        if not os.path.isfile(mesh_path_3):
            try:
                _normal_tmp, _, _center_tmp = fit_scene_plane_normal_from_objects(
                    mesh_path,
                    points_per_mesh=5000,
                    ransac_dist_thresh=sim_cfg.get('dist_thresh', 0.01),
                    up_hint=np.array([0, 0, 1]),
                    low_percentile=sim_cfg.get('low_p', 5),
                    low_percentile_fallback=sim_cfg.get('low_p', 5) + 5,
                    horizontal_cos_thresh=sim_cfg.get('cos_thresh', 0.8),
                )
                normalize_scene_to_ground(mesh_path, _normal_tmp, _center_tmp)
                resolve_penetration(mesh_path)
            except Exception as e:
                print(f"[WARN] Normalization failed for {folder}: {e}, skipping.")
                continue
            if not os.path.isfile(mesh_path_3):
                print(f"[WARN] mesh_norm.obj still missing for {folder}, skipping.")
                continue

        info_path = os.path.join(mesh_path, "scene_norm_info.txt")
        if os.path.isfile(info_path):
            with open(info_path, "r") as f:
                z_min_global = float(f.read().strip())

        c        = obj_cfg.get(idx) or obj_cfg.get(str(idx), {})
        mat_type = c.get('material', 'rigid')
        mat_params = c.get('material_params', {})
        material = get_material(mat_type, mat_params)

        surface_kwargs = {}
        if mat_type in SURFACE_DEFAULTS:
            default_color = SURFACE_DEFAULTS[mat_type]
            surface_color = tuple(c.get('surface_color', default_color))
            vis_mode = c.get('vis_mode', 'particle')
            surface_kwargs["surface"] = gs.surfaces.Default(
                color    = surface_color,
                vis_mode = vis_mode,
            )

        entity = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=mesh_path_3,
                pos=(c.get('x_off', 0), c.get('y_off', 0), c.get('z_off', 0)),
                euler=(0, 0, 0),
                scale = 1,
                fixed=c.get('fixed', False),
            ),
            material=material,
            **surface_kwargs,
        )

        entities[idx] = {
            "entity":      entity,
            "type":        mat_type,
            "vel":         c.get('velocity', [0, 0, 0, 0, 0, 0] if mat_type == "rigid" else [0, 0, 0]),
            "start_frame": c.get('start_frame', -1),
            "fix_top_ratio": c.get('fix_top_ratio', 0.0),
        }


    print(f"[INFO] Camera alignment mode: {camera_align}")

    if camera_align == "new":
        try:
            pos2, lookat2, cam_fov, align_method = _get_camera_new(
                image_path=image_path,
                mask_dir=mask_dir,
                mesh_path=mesh_path,
                sim_cfg=sim_cfg,
                z_min_global=z_min_global,
                device=device,
            )
            print(f"[INFO] Camera pose from [{align_method}]: pos={pos2}, lookat={lookat2}, fov={cam_fov:.1f}")
        except Exception as e:
            print(f"[WARN] New camera align failed ({e}), falling back to legacy.")
            pos2, lookat2, cam_fov, align_method, _, _ = _get_camera_legacy(
                mesh_path, sim_cfg, z_min_global
            )
    else:
        pos2, lookat2, cam_fov, align_method, _, _ = _get_camera_legacy(
            mesh_path, sim_cfg, z_min_global
        )
        print(f"[INFO] Camera pose from [{align_method}]: pos={pos2}, lookat={lookat2}, fov={cam_fov:.1f}")

    # ---- Camera fine-tuning (shared by both methods) ----
    pos2 = move_pos_along_view(pos2, lookat2, sim_cfg.get('camera_tre', 0))
    pos2[1] += sim_cfg.get('posy_off', 0)
    pos2[2] += sim_cfg.get('posz_off', 0)

    # ---- Load images ----
    target = cv2.imread(image_path)
    original_h, original_w = target.shape[:2]
    target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
    # target = cv2.resize(target, (880, 880), interpolation=cv2.INTER_AREA)

    bg = cv2.imread(f"{BASE_PATH}/inpaint/inpaint_all.png")
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    bg = cv2.resize(bg, (original_w, original_h), interpolation=cv2.INTER_AREA)
    h, w, _ = bg.shape

    # ---- Camera (using selected fov) ----
    cam = scene.add_camera(
        res=(w, h),
        fov=cam_fov,
        pos=pos2,
        lookat=lookat2,
        GUI=False,
    )

    # ---- Force fields (must be before scene.build()) ----
    force_cfgs = cfg.get('forces', [])
    force_fields = []
    for fc in (force_cfgs or []):
        ff = get_force_field(fc)
        scene.add_force_field(ff)
        force_fields.append((ff, fc))

    scene.build()

    # ---- Fix top particles for PBD cloth entities (e.g. dress) ----
    # for idx, data in entities.items():
    #     fix_ratio = data.get("fix_top_ratio", 0.0)
    #     if fix_ratio > 0 and "pbd_cloth" in data["type"]:
    #         ent = data["entity"]
    #         pos = ent.get_particles_pos()            # (material, N, 3) or (N, 3)
    #         if pos.dim() == 3:
    #             pos = pos[0]                         # (N, 3)
    #         z_vals = pos[:, 2]                       # z is up
    #         n_fix = max(1, int(fix_ratio * len(z_vals)))
    #         top_indices = z_vals.argsort(descending=True)[:n_fix]
    #         ent.fix_particles(particles_idx_local=top_indices.cpu().tolist())
    #         print(f"[INFO] Fixed top {n_fix}/{len(z_vals)} particles for object {idx} (ratio={fix_ratio})")

    # ---- Activate force fields: start_frame < 0 means immediate ----
    for ff, fc in force_fields:
        if fc.get('active', True) and fc.get('start_frame', -1) < 0:
            ff.activate()

    # ---- Camera position + FOV optimisation ----
    x0 = np.array([pos2[0], pos2[1], pos2[2], cam_fov], dtype=np.float64)

    bounds = [
        (x0[0] - sim_cfg.get("opt_dx", 0.5), x0[0] + sim_cfg.get("opt_dx", 0.5)),
        (x0[1] - sim_cfg.get("opt_dy", 0.5),  x0[1] + sim_cfg.get("opt_dy", 0.5)),
        (x0[2] - sim_cfg.get("opt_dz", 0.5),  x0[2] + sim_cfg.get("opt_dz", 0.5)),
        (max(20, cam_fov - sim_cfg.get("opt_dfov", 25)), min(120, cam_fov + sim_cfg.get("opt_dfov", 25))),
    ]

    from utils.util_camera import make_objective, random_search, refine

    t_opt0 = time.perf_counter()

    obj = make_objective(
        cam, bg, target,
        lookat0=lookat2,
        image_mask_path=image_mask_path,
        w_obj=sim_cfg.get("w_obj", 1.0),
        w_bg=sim_cfg.get("w_bg", 0.2),
        w_mask=sim_cfg.get("w_mask", 1.0),
    )

    x_coarse, f_coarse = random_search(
        obj, x0, bounds,
        n=sim_cfg.get("opt_rand_n", 100),
        seed=0,
    )
    x_best, f_best = refine(obj, x_coarse, bounds)

    t_opt1 = time.perf_counter()

    x, y, z, fov_opt = [float(t) for t in x_best]
    pos2 = [x, y, z]
    cam.set_pose(pos=pos2, lookat=lookat2)
    cam._fov = fov_opt

    # Save first-frame comparison: original | rendered
    final_check = render_with_clean_shadows(cam, bg)
    os.makedirs(output_dir, exist_ok=True)
    compare = np.hstack([target, final_check])
    cv2.imwrite(
        f"{output_dir}/{scene_name}_compare.png",
        cv2.cvtColor(compare, cv2.COLOR_RGB2BGR),
    )

    print(f"[INFO] Optimized pos={pos2}, fov={fov_opt:.1f}, loss={f_best:.6f}, time={t_opt1 - t_opt0:.1f}s")
    print(f"[INFO] Comparison saved: {output_dir}/{scene_name}_compare.png")

    # ---- Simulation loop ----
    t_sim0 = time.perf_counter()

    cam.start_recording()

    v, dt_cam = 0.001, 1.0 / 60
    mv = move
    print(f"[INFO] Camera move type: {mv}")

    n_steps = sim_cfg.get('n_steps', 300)

    for i in range(n_steps):
        scene.step()

        # Per-frame force activation
        for ff, fc in force_fields:
            if i == fc.get('start_frame', -1):
                ff.activate()

        for idx, data in entities.items():
            if i == data["start_frame"]:
                ent = data["entity"]
                vel = data["vel"]
                if "rigid" in data["type"]:
                    ent.set_dofs_velocity(vel)
                else:
                    ent.set_particles_vel(vel)

        # Camera motion
        if mv in [1, 2, 3, 4]:
            axis_indices = [0, 1] if mv in [1, 2] else [1, 2]
            R_cam  = np.sqrt(pos2[axis_indices[0]]**2 + pos2[axis_indices[1]]**2)
            theta0 = np.arctan2(pos2[axis_indices[1]], pos2[axis_indices[0]])
            direction = 1 if mv in [1, 3] else -1
            theta = theta0 + direction * (v / R_cam) * i * dt_cam
            pos2[axis_indices[0]] = R_cam * np.cos(theta)
            pos2[axis_indices[1]] = R_cam * np.sin(theta)
            cam.set_pose(pos=pos2)
        elif mv == 5:
            pos2[1] += i * v * dt_cam
            cam.set_pose(pos=pos2)
        elif mv == 6:
            pos2[2] -= i * v * dt_cam
            cam.set_pose(pos=pos2)

        final = render_with_clean_shadows(cam, bg)
        # final = render_with_objects(cam)


        cam._recorded_imgs[-1] = final

    t_sim1 = time.perf_counter()
    print(f"[TIMING] simulation_loop_sec={t_sim1 - t_sim0:.1f} (steps={n_steps})")
    print(f"[TIMING] camera_optimization_sec={t_opt1 - t_opt0:.1f}")
    print(f"[TIMING] camera_align_method={align_method}")

    os.makedirs(output_dir, exist_ok=True)
    cam.stop_recording(
        f"{output_dir}/{scene_name}.mp4",
        fps=sim_cfg.get('fps', 60),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",      type=str,  default=None)
    parser.add_argument("--scene_name",    type=str,  default="horse")
    parser.add_argument("--output_dir",    type=str,  default="demo/demo_plus")
    parser.add_argument("--move",          type=int,  default=0)
    parser.add_argument("--config",        type=str,  default=None,
                        help="Optional path to config.yaml")
    parser.add_argument(
        "--camera_align",
        type=str,
        default="legacy",
        choices=["legacy", "new"],
        help=(
            "Camera alignment mode:\n"
            "  legacy  Old method: fit ground normal + heuristic pose (default)\n"
            "  new     New method: 3-tier alignment (DA3 prior -> PnP -> heuristic)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for new camera alignment (only used when --camera_align=new)",
    )

    args = parser.parse_args()

    run_scene(
        args.root_dir,
        args.scene_name,
        config=args.config,
        output_dir=args.output_dir,
        move=args.move,
        camera_align=args.camera_align,
        device=args.device,
    )
