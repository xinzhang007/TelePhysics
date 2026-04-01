"""Multi-GPU video synthesis: scenes × prompts × steps on N GPUs.

Launches parallel workers across all available GPUs. Each worker processes
one (scene, prompt, steps) job using the Wan2.2-VACE pipeline.

Usage:
    # Use all 8 GPUs, all prompts, steps=10,30,50 (default)
    python pipeline/video_synthesis_batch.py

    # Specify scenes / GPUs / steps
    python pipeline/video_synthesis_batch.py --gpus 0,1,2,3 --scenes ball1,cream --steps 10,50
"""

import argparse
import os
import sys
import glob
import time
import torch
import torch.multiprocessing as mp
from PIL import Image

# ── Prompts ──────────────────────────────────────────────────────────────
PROMPTS = {
    "cinematic": (
        "遵循现有物理轨迹和结构运动，提升场景的视觉质量。"
        "增强电影级光照效果，包括柔和的阴影和全局照明。"
        "高分辨率 4K，逼真的纹理，专业的视觉特效渲染,背景不变。"
    ),
    "dramatic": (
        "Follow the existing physical trajectory and structural motion. "
        "Apply dramatic cinematic lighting with deep shadows, volumetric fog, "
        "and intense color grading. Slow motion feel, ultra-detailed 4K textures, "
        "film grain, professional VFX compositing. Background stays the same."
    ),
    "stylized": (
        "Follow the existing physics motion faithfully. "
        "Render in a hyper-realistic style with vivid saturated colors, "
        "sharp focus, studio-quality lighting with rim light highlights. "
        "8K detail, photorealistic material rendering, clean background preserved."
    ),
    "retexture": (
        "视频展示了物体在真实物理模拟下的运动轨迹与形变过程。"
        "请严格参照参考图片中物体的原始纹理、材质、颜色和细节，"
        "将真实外观精确地还原并贴合到每一帧中运动的物体表面上。"
        "物体表面必须保持与参考图片一致的质感和光泽，纹理清晰锐利，"
        "光照和阴影随物理运动自然变化。"
        "背景保持与参考图片完全一致，不做任何修改或风格化处理。"
        "最终效果应如同将参考图片中的真实物体放入物理世界中运动，"
        "呈现照片级真实感，无任何人工痕迹。"
    ),
}

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def worker_fn(gpu_id: int, job_queue, root_dir: str, base_output: str, output_dir: str):
    """Worker: pulls jobs from the queue, runs Wan2.2-VACE synthesis."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from diffsynth.utils.data import save_video, VideoData
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from utils.myutils import uniform_sample_video

    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    print(f"[GPU {gpu_id}] Loading Wan2.2-VACE pipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B",
                        origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                        **vram_config),
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B",
                        origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                        **vram_config),
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        **vram_config),
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B",
                        origin_file_pattern="Wan2.1_VAE.pth",
                        **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                                     origin_file_pattern="google/umt5-xxl/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
    )
    print(f"[GPU {gpu_id}] Pipeline loaded.")

    os.makedirs(output_dir, exist_ok=True)
    h = w = 880

    while True:
        try:
            job = job_queue.get_nowait()
        except Exception:
            break

        scene_name, prompt_name, prompt_text, n_steps = job
        scene_output_dir = os.path.join(base_output, scene_name)
        depth_dir = os.path.join(scene_output_dir, "depth")
        tmp_dir = os.path.join(scene_output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        final_path = os.path.join(output_dir, f"{scene_name}_{prompt_name}_s{n_steps}.mp4")
        if os.path.exists(final_path):
            print(f"[GPU {gpu_id}] SKIP {scene_name}/{prompt_name}/s{n_steps}: already exists")
            continue

        depth_video = os.path.join(depth_dir, f"{scene_name}.mp4")
        if not os.path.exists(depth_video):
            print(f"[GPU {gpu_id}] SKIP {scene_name}: no depth video found")
            continue

        image_path = os.path.join(root_dir, scene_name, f"{scene_name}.png")
        if not os.path.exists(image_path):
            print(f"[GPU {gpu_id}] SKIP {scene_name}: no scene image found")
            continue

        print(f"[GPU {gpu_id}] START {scene_name} / {prompt_name} / steps={n_steps}")
        t0 = time.time()

        # Prepare depth video
        processed_path = os.path.join(tmp_dir, f"{scene_name}_{prompt_name}_prepared.mp4")
        video_path_ready = uniform_sample_video(
            input_path=depth_video,
            num_frames=81,
            target_fps=15,
            target_size=(h, w),
            output_path=processed_path,
        )
        control_video = VideoData(video_path_ready, height=h, width=w)

        video = pipe(
            prompt=prompt_text,
            negative_prompt=NEGATIVE_PROMPT,
            seed=1,
            tiled=True,
            vace_video=control_video,
            vace_reference_image=Image.open(image_path).convert("RGB").resize((h, w)),
            denoising_strength=1,
            num_inference_steps=n_steps,
            height=h,
            width=w,
        )

        save_video(video, final_path, fps=15, quality=5)
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"[GPU {gpu_id}] DONE {scene_name}/{prompt_name}/s{n_steps} ({elapsed:.1f}s) -> {final_path}")

    print(f"[GPU {gpu_id}] Worker finished, no more jobs.")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU batch video synthesis")
    parser.add_argument("--root_dir", type=str, default="data/fluid",
                        help="Root directory containing scene folders")
    parser.add_argument("--base_output", type=str, default="demo/output_fluid",
                        help="Base output directory (contains {scene}/depth/)")
    parser.add_argument("--output_dir", type=str, default="demo/output_fluid/wan",
                        help="Single output directory for all mp4 files")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (default: all available)")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene names (default: all)")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Comma-separated prompt keys (default: all)")
    parser.add_argument("--steps", type=str, default="10,30,50",
                        help="Comma-separated num_inference_steps values (default: 10,30,50)")
    args = parser.parse_args()

    # Determine GPUs
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    n_gpus = len(gpu_ids)
    print(f"Using {n_gpus} GPUs: {gpu_ids}")

    # Determine scenes
    if args.scenes:
        scenes = args.scenes.split(",")
    else:
        scenes = sorted([
            d for d in os.listdir(args.root_dir)
            if os.path.isdir(os.path.join(args.root_dir, d))
        ])
    print(f"Scenes: {scenes}")

    # Determine prompts
    if args.prompts:
        prompt_keys = args.prompts.split(",")
    else:
        prompt_keys = list(PROMPTS.keys())
    print(f"Prompts: {prompt_keys}")

    # Determine steps
    steps_list = [int(s) for s in args.steps.split(",")]
    print(f"Steps: {steps_list}")

    # Use spawn context for CUDA compatibility
    ctx = mp.get_context("spawn")

    # Build job queue: (scene, prompt_name, prompt_text, n_steps)
    job_queue = ctx.Queue()
    n_jobs = 0
    for scene in scenes:
        for pk in prompt_keys:
            if pk not in PROMPTS:
                print(f"[WARN] Unknown prompt key: {pk}, skipping")
                continue
            for ns in steps_list:
                job_queue.put((scene, pk, PROMPTS[pk], ns))
                n_jobs += 1

    print(f"Total jobs: {n_jobs} ({len(scenes)} scenes × {len(prompt_keys)} prompts × {len(steps_list)} steps)")
    print(f"Distributing across {n_gpus} GPUs\n")

    # Launch workers
    processes = []
    for gpu_id in gpu_ids:
        p = ctx.Process(
            target=worker_fn,
            args=(gpu_id, job_queue, args.root_dir, args.base_output, args.output_dir),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n========== All synthesis jobs completed ==========")
    print(f"Outputs: {args.output_dir}/{{scene}}_{{prompt}}_s{{steps}}.mp4")


if __name__ == "__main__":
    main()
