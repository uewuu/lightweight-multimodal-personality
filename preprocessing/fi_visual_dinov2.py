import os
import sys
import argparse
import time
import csv
from pathlib import Path

# ================= 路径设置 =================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# 让 Python 优先搜索当前 preprocess 目录，便于导入 dinov2_face.py
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(PROJECT_ROOT))

# 切换到项目根目录：<PROJECT_ROOT>
os.chdir(PROJECT_ROOT)

import torch
import numpy as np

from exordium.video.facedetector import RetinaFaceDetector
from exordium.video.tracker import IouTracker
from exordium.video.detection import FrameDetections, VideoDetections
from exordium.utils.device import get_device_str
from dinov2_face import DinoV2FaceWrapper

# ================= 数据路径 =================
DB = Path("datas")
DB_VIDEOS = DB / "videos"

def collect_video_paths(video_root: Path, pattern: str) -> list[Path]:
    """Collect videos under data/db_processed/fi/videos."""
    video_paths = sorted(list(video_root.glob(pattern)))

    if len(video_paths) == 0:
        print("Warning: No files found by pattern. Trying common video extensions...")
        video_paths = []
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4"]:
            video_paths.extend(list(video_root.glob(f"**/{ext}")))
        video_paths = sorted(list(set(video_paths)))

    return video_paths


def _load_vdet_csv(tracker_path: Path, video_path: Path) -> VideoDetections:
    """Load exordium .vdet CSV file into a VideoDetections object.

    The existing FI tracker files are CSV-like text files, not pickle files.

    Note: older .vdet files may store stale source paths such as
    data/db_processed/fi/videos/..., which do not match the current datas/videos layout.
    Therefore, this loader overrides each detection source with the current video_path.
    Expected columns:
        frame_id, source, score, x, y, w, h,
        left_eye_x, left_eye_y, right_eye_x, right_eye_y,
        nose_x, nose_y, left_mouth_x, left_mouth_y,
        right_mouth_x, right_mouth_y
    """
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker file does not exist: {tracker_path}")

    video_detections = VideoDetections()

    with open(tracker_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {
            "frame_id", "source", "score", "x", "y", "w", "h",
            "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
            "nose_x", "nose_y", "left_mouth_x", "left_mouth_y",
            "right_mouth_x", "right_mouth_y",
        }
        missing_cols = required_cols - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Invalid .vdet format. Missing columns: {sorted(missing_cols)}")

        for row in reader:
            # Some .vdet files may contain blank lines.
            if not row or row.get("frame_id") in (None, ""):
                continue

            frame_id = int(float(row["frame_id"]))

            # The original detector stores bb_xywh in the FrameDetections object.
            bb_xywh = np.array([
                int(float(row["x"])),
                int(float(row["y"])),
                int(float(row["w"])),
                int(float(row["h"])),
            ], dtype=int)

            # Keep the same semantic order as the CSV header.
            # This is sufficient for tracking/cropping because the bbox is the key field.
            landmarks = np.array([
                [int(float(row["left_eye_x"])), int(float(row["left_eye_y"]))],
                [int(float(row["right_eye_x"])), int(float(row["right_eye_y"]))],
                [int(float(row["nose_x"])), int(float(row["nose_y"]))],
                [int(float(row["left_mouth_x"])), int(float(row["left_mouth_y"]))],
                [int(float(row["right_mouth_x"])), int(float(row["right_mouth_y"]))],
            ], dtype=int)

            frame_detections = FrameDetections()
            frame_detections.add_dict({
                "frame_id": frame_id,
                "source": str(video_path),
                "score": float(row["score"]),
                "bb_xywh": bb_xywh,
                "landmarks": landmarks,
            })
            video_detections.add(frame_detections)

    return video_detections


def build_or_load_track(
    video_path: Path,
    tracker_path: Path,
    face_detector: RetinaFaceDetector,
    force_redetect: bool = False,
):
    """Build the face track for DINOv2 extraction.

    This version prefers existing CSV .vdet files and does NOT rerun RetinaFace
    unless the tracker file is missing or --force_redetect is explicitly used.
    """
    tracker_path.parent.mkdir(parents=True, exist_ok=True)

    if tracker_path.exists() and not force_redetect:
        print(f"  [LOAD] Loading existing tracker detections: {tracker_path}")
        videodetections = _load_vdet_csv(tracker_path, video_path)
        print("  [OK] Existing CSV .vdet loaded successfully")
    else:
        print("  [DETECT] Detecting faces...")
        videodetections = face_detector.detect_video(
            video_path,
            output_path=tracker_path,
        )

    print("  [TRACK] Tracking...")
    track = (
        IouTracker(max_lost=30)
        .label(videodetections)
        .merge()
        .select_topk_biggest_bb_tracks(top_k=2)
        .select_topk_long_tracks(top_k=2)
        .get_center_track()
    )
    return track


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2/ViT face features for ChaLearn FI without re-extracting OpenGraphAU/FABNet."
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--start", type=int, default=0, help="Video slice start index.")
    parser.add_argument("--end", type=int, default=10000, help="Video slice end index.")
    parser.add_argument("--pattern", type=str, default="**/*.mp4", help="Video glob pattern.")
    parser.add_argument("--batch_size", type=int, default=30, help="DINOv2 extraction batch size.")
    parser.add_argument(
        "--force_redetect",
        action="store_true",
        help="Force face detection even if tracker .vdet exists.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DINOv2 Face Feature Extraction")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Video root:   {DB_VIDEOS}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name:  {torch.cuda.get_device_name(args.gpu_id)}")

    device = get_device_str(args.gpu_id)
    print(f"get_device_str({args.gpu_id}) = {device}")
    print("=" * 60)

    print("Initializing face detector...")
    face_detector = RetinaFaceDetector(gpu_id=args.gpu_id, batch_size=10)

    print("Initializing DINOv2 extractor...")
    dinov2_extractor = DinoV2FaceWrapper(gpu_id=args.gpu_id)

    print(f"Searching videos with pattern: {args.pattern}")
    video_paths = collect_video_paths(DB_VIDEOS, args.pattern)
    video_paths = video_paths[args.start:args.end]

    print(f"Number of selected videos: {len(video_paths)}")
    if len(video_paths) == 0:
        raise FileNotFoundError(f"No video files found in {DB_VIDEOS.resolve()}")

    skip_log = DB / "fi_skip_dinov2_face.txt"
    skip_log.parent.mkdir(parents=True, exist_ok=True)

    for idx, video_path in enumerate(video_paths, start=1):
        start_time = time.time()

        video_name = video_path.parent.name
        video_id = video_path.stem

        tracker_path = DB / "tracker" / video_name / f"{video_id}.vdet"
        dinov2_path = DB / "dinov2_face" / video_name / f"{video_id}.pkl"

        progress = f"[{idx}/{len(video_paths)}] {idx / len(video_paths) * 100:.1f}%"
        print(f"\n{progress} Processing: {video_name}/{video_id}")

        # 核心保护：已有 DINOv2 特征则跳过，不碰任何旧特征。
        if dinov2_path.exists():
            print(f"  [SKIP] DINOv2 feature already exists, skipping: {dinov2_path}")
            continue

        try:
            track = build_or_load_track(
                video_path=video_path,
                tracker_path=tracker_path,
                face_detector=face_detector,
                force_redetect=args.force_redetect,
            )

            print(f"  [OK] Track length: {len(track)}")

            print("  [DINOv2] Extracting DINOv2 face features...")
            dinov2_path.parent.mkdir(parents=True, exist_ok=True)
            ids, features = dinov2_extractor.track_to_feature(
                track,
                batch_size=args.batch_size,
                output_path=dinov2_path,
            )

            print(f"  [OK] DINOv2 feature shape: {features.shape}")
            print(f"  [SAVE] Saved to: {dinov2_path}")

            elapsed = time.time() - start_time
            print(f"  [TIME] Completed in {elapsed:.2f}s")

        except Exception as e:
            print(f"  [ERROR] Error: {e}")
            with open(skip_log, "a", encoding="utf-8") as f:
                f.write(f"{video_name} | {video_id} | {e}\n")


if __name__ == "__main__":
    main()
