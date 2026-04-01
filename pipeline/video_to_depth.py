import argparse
import numpy as np
import os
import torch
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils_vda import read_video_frames, save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    # Add root_dir and scene_name arguments
    parser.add_argument('--root_dir', type=str, default='demo/demo_test', help="Root directory for the data")
    parser.add_argument('--scene_name', type=str, default='cup', help="Scene name for processing")
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--metric', action='store_true', help='use metric model')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')
    parser.add_argument("--move", type=int, default = 0)

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # import pdb; pdb.set_trace()

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    # Initialize the model
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder], metric=args.metric)
    video_depth_anything.load_state_dict(torch.load(f'./models/VDA/{checkpoint_name}_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # Construct the path for the specific video using root_dir and scene_name
    video_path = os.path.join(args.root_dir, f'{args.scene_name}.mp4')
    
    # Make the output directory if it doesn't exist
    output_dir = os.path.join(args.root_dir, 'depth')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing video: {video_path}")
    
    # Read video frames
    frames, target_fps = read_video_frames(video_path, args.max_len, args.target_fps, args.max_res)
    
    # Perform depth inference
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

    # Extract the scene name to use in output filenames
    video_name = args.scene_name

    # Save depth visualization as a new video
    depth_vis_path = os.path.join(output_dir, f'{video_name}.mp4')
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=True)

    print("Processing completed for the video.")
