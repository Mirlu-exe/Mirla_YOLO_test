import os
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

import sys
from tqdm import tqdm

# -------------------------------------------------------------------------------------------------------
#
#  YOLOv11 Tracking Version — Adds object tracking and merges all chunks into one final video.
#  Works with your custom elephant model or pretrained YOLO model.
#
# -------------------------------------------------------------------------------------------------------

# -----------------------------
# Config (tune these)
# -----------------------------
VIDEOS_DIR = Path('./Mirla_YOLO_test_videos/unseen_videos/Target')
INPUT_VIDEO = VIDEOS_DIR / '0820_8.30session_site2_1_h264.mp4'
OUTPUT_VIDEO = INPUT_VIDEO.with_name(INPUT_VIDEO.stem + '_tracked.mp4')

## Custom model (fine-tuned elephants)
MODEL_PATH = Path('./runs/detect/train21/weights/last.pt')

CHUNK_SECONDS = 180            # process in N-second chunks
TARGET_HEIGHT = 720            # resize height (keeps aspect ratio)
CONF_THRESHOLD = 0.5
YOLO_IMG_SIZE = 1280            # YOLO internal inference size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FFMPEG = 'ffmpeg'
FFPROBE = 'ffprobe'

# tqdm options
_TQDM_KW = dict(dynamic_ncols=True, ascii=False, disable=not sys.stderr.isatty())

# -----------------------------
# Helper functions
# -----------------------------
def probe_duration_seconds(video_path: Path) -> float:
    """Try ffprobe; fallback to OpenCV."""
    try:
        out = subprocess.check_output(
            [FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
            stderr=subprocess.STDOUT
        ).decode().strip()
        dur = float(out)
        if dur > 0:
            return dur
    except Exception:
        pass

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps > 0 and frames > 0:
        return frames / fps
    raise RuntimeError("Could not determine video duration via ffprobe or OpenCV.")


def probe_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and not math.isclose(fps, 0.0) else 30.0


def make_resized_chunk(src: Path, dst: Path, start_s: float, dur_s: float, target_h: int):
    """Use ffmpeg to cut and scale in one pass."""
    cmd = [
        FFMPEG, '-y',
        '-ss', f'{start_s:.3f}',
        '-t', f'{dur_s:.3f}',
        '-i', str(src),
        '-vf', f'scale=-2:{target_h}',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
        '-an',
        str(dst)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# -----------------------------
# Prepare model and inputs
# -----------------------------
if not INPUT_VIDEO.exists():
    raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

duration = probe_duration_seconds(INPUT_VIDEO)
fps = probe_fps(INPUT_VIDEO)

# Load YOLO model
model = YOLO(str(MODEL_PATH))
model.to(DEVICE)
names = model.names

tqdm.write(f"[INFO] Duration: {duration:.2f}s | FPS: {fps:.3f} | Device: {DEVICE}")
tqdm.write(f"[INFO] Target height: {TARGET_HEIGHT}px; YOLO imgsz: {YOLO_IMG_SIZE}; conf: {CONF_THRESHOLD}")

# -----------------------------
# Tracking setup
# -----------------------------
TRACK_CONF = 0.5
PERSIST = True  # Keep track IDs across frames


def track_and_write(video_path, output_path):
    """Use YOLOv11 tracking (ByteTrack) to process the video."""
    results = model.track(
        source=str(video_path),
        conf=TRACK_CONF,
        imgsz=YOLO_IMG_SIZE,
        device=DEVICE,
        persist=PERSIST,
        stream=True,
        verbose=False
    )

    out_writer = None
    for result in results:
        frame = result.plot()  # Draw boxes, labels, and track IDs
        if out_writer is None:
            H, W = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
        out_writer.write(frame)

    if out_writer is not None:
        out_writer.release()


# -----------------------------
# Chunked tracking + merge
# -----------------------------
tmpdir = Path(tempfile.mkdtemp(prefix='yolo_chunks_'))

try:
    num_chunks = int(math.ceil(duration / CHUNK_SECONDS))
    start = 0.0
    chunk_pbar = tqdm(total=num_chunks, desc='Chunks', position=0, leave=True, **_TQDM_KW)

    for chunk_idx in range(num_chunks):
        remaining = max(0.0, duration - start)
        this_len = min(CHUNK_SECONDS, remaining)
        if this_len <= 0:
            break

        chunk_path = tmpdir / f'chunk_{chunk_idx:04d}.mp4'
        tracked_chunk_path = tmpdir / f'tracked_{chunk_idx:04d}.mp4'

        # Cut + resize chunk
        make_resized_chunk(INPUT_VIDEO, chunk_path, start, this_len, TARGET_HEIGHT)

        tqdm.write(f"[INFO] Tracking chunk {chunk_idx+1}/{num_chunks} ({this_len:.1f}s)")
        track_and_write(chunk_path, tracked_chunk_path)

        try:
            chunk_path.unlink(missing_ok=True)
        except Exception:
            pass

        if DEVICE.startswith('cuda'):
            torch.cuda.empty_cache()

        chunk_pbar.update(1)
        start += this_len

    chunk_pbar.close()

    # Merge tracked chunks
    concat_list = tmpdir / 'concat.txt'
    with open(concat_list, 'w') as f:
        for i in range(num_chunks):
            f.write(f"file '{tmpdir / f'tracked_{i:04d}.mp4'}'\n")

    tqdm.write("[INFO] Merging tracked chunks into final output...")
    subprocess.run([
        FFMPEG, '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_list),
        '-c', 'copy', str(OUTPUT_VIDEO)
    ], check=True)

    tqdm.write(f"[DONE] Final tracked video saved as: {OUTPUT_VIDEO}")

finally:
    cv2.destroyAllWindows()
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass
