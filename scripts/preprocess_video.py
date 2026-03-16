"""
Extract and preprocess frames from video files.

Usage:
    python scripts/preprocess_video.py \
        --input_dir data/raw_video/ \
        --output_dir data/video/ \
        --fps 30
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_frames(video_path: Path, output_dir: Path, fps: int = 30, segment_sec: float = 3.0):
    """Extract 3-second segments from a video file as JPEG frames.

    Each segment becomes a subdirectory:
        output_dir / <video_stem> / seg_<N> / frame_<F>.jpg
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frames_per_seg = int(segment_sec * video_fps)
    stem = video_path.stem

    seg_idx   = 0
    frame_idx = 0
    seg_dir   = output_dir / stem / f"seg_{seg_idx:04d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = seg_dir / f"frame_{seg_frame:04d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        seg_frame += 1
        frame_idx += 1

        if seg_frame >= frames_per_seg:
            seg_idx  += 1
            seg_frame = 0
            seg_dir   = output_dir / stem / f"seg_{seg_idx:04d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

    cap.release()
    return seg_idx + 1


def main():
    parser = argparse.ArgumentParser(description="Preprocess video files to frames")
    parser.add_argument("--input_dir",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps",        type=int, default=30)
    parser.add_argument("--overwrite",  action="store_true")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    print(f"Found {len(video_files)} video files")

    total_segments = 0
    for video_path in tqdm(video_files):
        stem = video_path.stem
        dest = output_dir / stem
        if not args.overwrite and dest.exists() and any(dest.iterdir()):
            continue
        n = extract_frames(video_path, output_dir, fps=args.fps)
        total_segments += n

    print(f"Extracted {total_segments} segments total.")


if __name__ == "__main__":
    main()
