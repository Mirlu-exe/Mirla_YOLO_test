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

# -----------------------------
# Config (tune these)
# -----------------------------
VIDEOS_DIR = Path('./unseen_videos/Target')
INPUT_VIDEO = VIDEOS_DIR / '0820_8.30session_site2_1_h264.mp4'
OUTPUT_VIDEO = INPUT_VIDEO.with_name(INPUT_VIDEO.stem + '_out.mp4')

    ## Pretrained model
#MODEL_PATH = Path('yolo11s.pt')

    ## Custom model
MODEL_PATH = Path('./runs/detect/train13/weights/last.pt')


CHUNK_SECONDS = 180            # process in N-second chunks
TARGET_HEIGHT = 720            # resize height (keeps aspect ratio)
CONF_THRESHOLD = 0.5
YOLO_IMG_SIZE = 640            # YOLO internal inference size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FFMPEG = 'ffmpeg'
FFPROBE = 'ffprobe'

# tqdm common options
_TQDM_KW = dict(dynamic_ncols=True, ascii=False, disable=not sys.stderr.isatty())

# -----------------------------
# Helpers
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
    """
    Use ffmpeg to cut and scale in one pass.
    scale = -2:target_h keeps aspect ratio and ensures even width for H.264.
    """
    cmd = [
        FFMPEG, '-y',
        '-ss', f'{start_s:.3f}',
        '-t', f'{dur_s:.3f}',
        '-i', str(src),
        '-vf', f'scale=-2:{target_h}',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
        '-an',  # no audio for temp chunk
        str(dst)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# -----------------------------
# Prepare
# -----------------------------
if not INPUT_VIDEO.exists():
    raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

duration = probe_duration_seconds(INPUT_VIDEO)
fps = probe_fps(INPUT_VIDEO)

# Load YOLO once
model = YOLO(str(MODEL_PATH))
model.to(DEVICE)

# temp dir for chunks
tmpdir = Path(tempfile.mkdtemp(prefix='yolo_chunks_'))

# OpenCV writer will be lazy-initialized on first frame
out_writer = None
names = model.names

try:
    num_chunks = int(math.ceil(duration / CHUNK_SECONDS))
    start = 0.0

    approx_total_frames = int(round(duration * fps))
    chunk_pbar   = tqdm(total=num_chunks, desc='Chunks', position=0, leave=True, **_TQDM_KW)
    overall_pbar = tqdm(total=approx_total_frames, desc='Frames (approx total)', position=1, leave=True, unit='f', **_TQDM_KW)

    tqdm.write(f"[INFO] Duration: {duration:.2f}s | FPS: {fps:.3f} | Chunks: {num_chunks} | Device: {DEVICE}")
    tqdm.write(f"[INFO] Target height: {TARGET_HEIGHT}px; YOLO imgsz: {YOLO_IMG_SIZE}; conf: {CONF_THRESHOLD}")

    def process_and_write(frm):
        results = model(frm, conf=CONF_THRESHOLD, imgsz=YOLO_IMG_SIZE, verbose=False, device=DEVICE)[0]
        if results and results.boxes is not None and len(results.boxes):
            for (x1, y1, x2, y2, score, class_id) in results.boxes.data.tolist():
                if score >= CONF_THRESHOLD: # for custom dataset trained model
                #if score >= CONF_THRESHOLD and int(class_id) == 20:  # 🐘 only elephants pretrained model
                    cv2.rectangle(frm, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    label = names.get(int(class_id), str(int(class_id)))
                    cv2.putText(frm, str(label).upper(), (int(x1), max(0, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        out_writer.write(frm)

    with torch.inference_mode():
        for chunk_idx in range(num_chunks):
            remaining = max(0.0, duration - start)
            this_len = min(CHUNK_SECONDS, remaining)
            if this_len <= 0:
                break

            chunk_path = tmpdir / f'chunk_{chunk_idx:04d}.mp4'

            # --- ffmpeg cut+scale progress (coarse: one step) ---
            ffmpeg_pbar = tqdm(total=1, desc=f'ffmpeg chunk {chunk_idx+1}', position=2, leave=False, **_TQDM_KW)
            make_resized_chunk(INPUT_VIDEO, chunk_path, start, this_len, TARGET_HEIGHT)
            ffmpeg_pbar.update(1)
            ffmpeg_pbar.close()

            # Open the resized chunk
            cap = cv2.VideoCapture(str(chunk_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open chunk: {chunk_path}")

            # Initialize writer on first frame
            ret, frame = cap.read()
            if not ret:
                cap.release()
                try: chunk_path.unlink(missing_ok=True)
                except Exception: pass
                chunk_pbar.update(1)
                start += this_len
                continue

            if out_writer is None:
                H, W = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (W, H))
                if not out_writer.isOpened():
                    cap.release()
                    raise RuntimeError(f"Failed to open output writer: {OUTPUT_VIDEO}")
                tqdm.write(f"[INFO] Output size: {W}x{H}")

            # Chunk frame progress (use real frame count if available)
            n_frames_chunk = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
            if n_frames_chunk > 0:
                frame_pbar = tqdm(total=n_frames_chunk, desc=f'Chunk {chunk_idx+1} frames', position=3, leave=False, unit='f', **_TQDM_KW)
            else:
                frame_pbar = tqdm(desc=f'Chunk {chunk_idx+1} frames', position=3, leave=False, unit='f', **_TQDM_KW)

            # process first frame
            process_and_write(frame)
            frame_pbar.update(1)
            overall_pbar.update(1)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                process_and_write(frame)
                frame_pbar.update(1)
                overall_pbar.update(1)

            cap.release()
            frame_pbar.close()

            # Delete chunk to keep disk usage bounded
            try:
                chunk_path.unlink(missing_ok=True)
            except Exception:
                pass

            # Reduce GPU fragmentation between chunks
            if DEVICE.startswith('cuda'):
                torch.cuda.empty_cache()

            chunk_pbar.update(1)
            start += this_len

    chunk_pbar.close()
    overall_pbar.close()

finally:
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

tqdm.write(f"[DONE] Wrote: {OUTPUT_VIDEO}")
