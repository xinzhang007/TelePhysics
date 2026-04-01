import torch
import os
import glob
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from PIL import Image
import argparse

from utils.myutils import uniform_sample_video

def process_videos(base_dir, output_dir):
    # Define directories
    image_dir = os.path.join(base_dir, "png")
    depth_dir = os.path.join(base_dir, "depth")
    output_dir1 = os.path.join(base_dir, "wan")  
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir1, exist_ok=True)

    # VRAM configuration
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

    # Initialize pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
    )

    # Get video files from the depth directory
    video_files = glob.glob(os.path.join(depth_dir, f"{args.scene_name}.mp4"))

    # Iterate over video files and process them
    for video_path in video_files:
        file_name = os.path.basename(video_path).replace(".mp4", "")
        if args.root_dir is not None:
            image_path = os.path.join(args.root_dir, args.scene_name, f"{args.scene_name}.png")
        else:
            image_path = os.path.join(image_dir, f"{file_name}.png")
        
        if not os.path.exists(image_path):
            print(f"Skipping {file_name}: Image not found at {image_path}")
            continue

        print(f"Processing: {file_name}...")
        orig_img = Image.open(image_path)
        orig_w, orig_h = orig_img.size
        # Round to nearest multiple of 8 for model compatibility
        w = (orig_w // 8) * 8
        h = (orig_h // 8) * 8

        # Prepare video
        processed_video_path = os.path.join(output_dir, f"{file_name}_prepared.mp4")
        video_path_ready = uniform_sample_video(
            input_path=video_path,
            num_frames=81,
            target_fps=15,
            target_size=(w, h),
            output_path=processed_video_path
        )

        control_video = VideoData(video_path_ready, height=h, width=w)

        # Generate video with the pipeline
        video = pipe(
            prompt="""
                遵循现有物理轨迹和结构运动，提升场景的视觉质量。增强电影级光照效果，包括柔和的阴影和全局照明。高分辨率 4K，逼真的纹理，专业的视觉特效渲染,背景不变。
            """,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            seed=1,
            tiled=True,
            vace_video=control_video,
            # vace_reference_image=Image.open(image_path).resize((h, w)),
            vace_reference_image = Image.open(image_path).convert("RGB").resize((w, h)),
            denoising_strength=1,
            num_inference_steps=args.num_inference_steps,
            height=h,
            width=w
        )

        # Save final output
        final_output_path = os.path.join(output_dir1, f"rendered_{file_name}.mp4")
        save_video(video, final_output_path, fps=15, quality=5)

        torch.cuda.empty_cache()

    print("All batch tasks completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process videos with the Wan Video Pipeline")
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default="demo/demo3", help="The base directory containing image and depth directories")
    parser.add_argument("--scene_name", type=str, default="apple", help="Scene name for processing")
    parser.add_argument("--output_dir", type=str, default="demo/tmp", help="The directory to save processed videos")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of denoising inference steps")


    args = parser.parse_args()

    process_videos(args.base_dir, args.output_dir)
