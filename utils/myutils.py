from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Tuple

def uniform_sample_video(
    input_path: str,
    num_frames: int = 81,
    target_fps: int = 15,
    target_size: Tuple[int, int] = (832, 480),  # (W, H)
    output_path: Optional[str] = None,
):
    """
    Uniformly sample frames from a video over the entire duration
    and save to a new video file.

    Args:
        input_path: path to input video
        num_frames: number of frames to sample (e.g. 81)
        target_fps: fps for output video
        target_size: (width, height)
        output_path: output video path (must be provided)

    Returns:
        output_path (str)
    """

    if output_path is None:
        raise ValueError("output_path must be provided")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Failed to get total frame count")

    # 均匀采样索引（覆盖首尾）
    indices = np.linspace(0, total_frames - 1, num_frames)
    indices = np.round(indices).astype(int)
    target_set = set(indices.tolist())

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, target_size)

    current_idx = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in target_set:
            frame = cv2.resize(frame, target_size)
            writer.write(frame)
            saved += 1
            if saved >= num_frames:
                break

        current_idx += 1

    cap.release()
    writer.release()

    if saved != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} frames, but got {saved}"
        )

    return output_path




import cv2
import numpy as np
from typing import Tuple, Optional


def uniform_sample_video_with_first_image(
    input_path: str,
    first_image_path: str,
    num_frames: int = 81,
    target_fps: int = 15,
    target_size: Tuple[int, int] = (832, 480),  # (W, H)
    output_path: Optional[str] = None,
):
    """
    Uniformly sample frames from a video, but replace the first frame
    with a given image.

    Total output frames = num_frames
    Frame 0            = first_image
    Frames 1~end       = uniformly sampled from video
    """

    if output_path is None:
        raise ValueError("output_path must be provided")

    if num_frames < 2:
        raise ValueError("num_frames must be >= 2")

    # ---------- Load first image ----------
    first_img = cv2.imread(first_image_path)
    if first_img is None:
        raise RuntimeError(f"Cannot read image: {first_image_path}")

    first_img = cv2.resize(first_img, target_size)

    # ---------- Open video ----------
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Failed to get total frame count")

    # ---------- Sample indices (num_frames - 1) ----------
    sample_count = num_frames - 1
    indices = np.linspace(0, total_frames - 1, sample_count)
    indices = np.round(indices).astype(int)
    target_set = set(indices.tolist())

    # ---------- Video writer ----------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, target_size)

    # ---------- Write first frame ----------
    writer.write(first_img)
    saved = 1

    # ---------- Read video and sample ----------
    current_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in target_set:
            frame = cv2.resize(frame, target_size)
            writer.write(frame)
            saved += 1
            if saved >= num_frames:
                break

        current_idx += 1

    cap.release()
    writer.release()

    if saved != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} frames, but got {saved}"
        )

    return output_path





def extract_first_frame(video_path: str, out_path: str = None):
    """
    提取视频首帧。
    - video_path: 视频路径
    - out_path: 若提供则把首帧保存为图片（png/jpg）
    返回: (frame, width, height)
      - frame: 首帧图像（BGR, numpy.ndarray）
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    ok, frame = cap.read()   # 读取第一帧
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("读取首帧失败（视频可能为空/损坏/编码不支持）")

    h, w = frame.shape[:2]

    if out_path:
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(out_path, frame):
            raise RuntimeError(f"首帧保存失败：{out_path}")

    return frame, w, h