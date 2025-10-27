import argparse
import sys
from pathlib import Path
import gc
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

"""
python /home/nishiyamalab/PycharmProjects/Mirla_YOLO_test/frame_extractor.py \
  --input_dir "/home/nishiyamalab/Videos/pinnawala_sample_videos" \
  --output_dir "/home/nishiyamalab/Videos/extracted_frames_pinnawala_sample_videos" \
  --clusters 12 \
  --sample_every 10 \
  --batch_size 512 \
  --resize_width 640 \
  --random_state 42
"""


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".MP4", ".MOV", ".M4V", ".WEBM", ".AVI", ".MKV"}

def list_videos(input_dir: Path):
    return [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in {e.lower() for e in VIDEO_EXTS} and p.is_file()]

def maybe_resize(img_bgr, resize_width: int | None):
    if not resize_width or resize_width <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if w <= resize_width:
        return img_bgr
    new_w = resize_width
    new_h = int(h * (new_w / w))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def hsv_histogram(img_bgr, bins=(8,8,8)):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0,1,2], None, bins, [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)

def laplacian_variance(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def grayscale_delta(prev_gray, curr_gray):
    if prev_gray is None:
        return 0.0
    mad = np.mean(np.abs(curr_gray.astype(np.int16) - prev_gray.astype(np.int16)))
    return float(mad) / 255.0

def frame_feature(img_bgr, prev_gray=None):
    hist = hsv_histogram(img_bgr)              # (512,)
    sharp = laplacian_variance(img_bgr)        # scalar
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    delta = grayscale_delta(prev_gray, gray)   # scalar
    feat = np.concatenate([hist, np.array([sharp, delta], dtype=np.float32)])  # (514,)
    return feat, gray

def sample_frames(cap, sample_every):
    """Yield tuples: (index, timestamp_ms, frame_bgr)."""
    idx = 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if frames <= 0:
        while True:
            ret = cap.grab()
            if not ret:
                break
            if idx % sample_every == 0:
                ret2, frame = cap.retrieve()
                if not ret2:
                    break
                ts_ms = int((idx / fps) * 1000)
                yield idx, ts_ms, frame
            idx += 1
        return

    for idx in range(0, frames, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        ts_ms = int((idx / fps) * 1000)
        yield idx, ts_ms, frame

def save_frames(selected, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    with tqdm(total=len(selected), desc=f"Saving {stem}", unit="img", leave=False) as pbar:
        for rank, (ts_ms, idx, frame_bgr) in enumerate(selected, start=1):
            filename = f"{stem}_rank{rank:02d}_f{idx:06d}_t{ts_ms:08d}ms.jpg"
            path = out_dir / filename
            cv2.imwrite(str(path), frame_bgr)
            pbar.update(1)

def train_kmeans_streaming(video_path: Path, sample_every: int, batch_size: int, resize_width: int | None, k: int, random_state: int):
    """Pass 1: stream features and partial_fit MiniBatchKMeans. No frames stored."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        tqdm.write(f"[WARN] Cannot open: {video_path}")
        return None

    km = MiniBatchKMeans(n_clusters=max(1, k), random_state=random_state, n_init=1, batch_size=batch_size)
    batch = []
    prev_gray = None

    # Progress estimation
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_samples_est = (frame_count + sample_every - 1) // sample_every if frame_count > 0 else None

    with tqdm(total=total_samples_est, desc=f"Pass1(train) {video_path.name}", unit="frame", leave=False) as pbar:
        for _, _, frame in sample_frames(cap, sample_every):
            frame = maybe_resize(frame, resize_width)
            feat, prev_gray = frame_feature(frame, prev_gray)
            batch.append(feat)

            if len(batch) >= batch_size:
                X = np.asarray(batch, dtype=np.float32)
                km.partial_fit(X)
                batch.clear()
                # occasional GC to keep RSS down
                gc.collect()

            pbar.update(1)

        if batch:
            X = np.asarray(batch, dtype=np.float32)
            km.partial_fit(X)
            batch.clear()
            gc.collect()

    cap.release()
    return km

def select_best_frames_streaming(video_path: Path, sample_every: int, resize_width: int | None, km: MiniBatchKMeans):
    """
    Pass 2: assign each sample to nearest centroid and keep only the best
    (closest) full-resolution frame per cluster.
    Returns list of tuples sorted by timestamp: (ts_ms, idx, frame_bgr).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        tqdm.write(f"[WARN] Cannot reopen: {video_path}")
        return []

    k = km.n_clusters
    # For each cluster, track (best_dist, ts_ms, idx, fullres_frame)
    best = [(np.inf, None, None, None) for _ in range(k)]

    prev_gray = None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_samples_est = (frame_count + sample_every - 1) // sample_every if frame_count > 0 else None

    with tqdm(total=total_samples_est, desc=f"Pass2(select) {video_path.name}", unit="frame", leave=False) as pbar:
        for idx, ts_ms, frame_full in sample_frames(cap, sample_every):
            # Compute feature on (optionally) downscaled frame to match Pass 1
            small = maybe_resize(frame_full, resize_width)
            feat, prev_gray = frame_feature(small, prev_gray)
            feat = feat.reshape(1, -1).astype(np.float32)

            # Assign to nearest centroid; use transform if available, else manual distance
            # MiniBatchKMeans has ._transform but public is .transform (euclidean to centers)
            dists = np.linalg.norm(km.cluster_centers_ - feat, axis=1)
            c = int(np.argmin(dists))
            d = float(dists[c])

            if d < best[c][0]:
                # Drop previous frame to free memory for this slot
                best[c] = (d, ts_ms, idx, frame_full)

            pbar.update(1)

    cap.release()

    # Collect chosen frames; discard empty clusters
    selected = [(ts, i, f) for (d, ts, i, f) in best if f is not None]
    # Sort by timeline for a nice spread
    selected.sort(key=lambda t: t[0])
    return selected

def process_video(video_path: Path, output_dir: Path, clusters: int, sample_every: int, random_state: int,
                  batch_size: int, resize_width: int | None):
    tqdm.write(f"[INFO] {video_path.name}: training k-means (k={clusters}) with streaming batches (size={batch_size})…")
    km = train_kmeans_streaming(video_path, sample_every, batch_size, resize_width, clusters, random_state)
    if km is None:
        return

    tqdm.write(f"[INFO] {video_path.name}: selecting best frames per cluster…")
    selected = select_best_frames_streaming(video_path, sample_every, resize_width, km)

    out_subdir = output_dir / video_path.stem
    tqdm.write(f"[INFO] {video_path.name}: saving {len(selected)} frames → {out_subdir}")
    save_frames(selected, out_subdir, stem=video_path.stem)
    tqdm.write(f"[OK] {video_path.name}: done.")

def main():
    parser = argparse.ArgumentParser(description="Memory-safe extraction of representative frames via streaming MiniBatchKMeans.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames.")
    parser.add_argument("--clusters", type=int, default=12, help="Number of frames to extract per video (K).")
    parser.add_argument("--sample_every", type=int, default=10, help="Sample every Nth frame (larger = faster).")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed for clustering.")
    parser.add_argument("--batch_size", type=int, default=512, help="MiniBatchKMeans batch size for partial_fit.")
    parser.add_argument("--resize_width", type=int, default=640, help="Optional downscale width before feature extraction (<=0 to disable).")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(in_dir)
    if not videos:
        print(f"[ERROR] No videos found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    tqdm.write(f"[INFO] Found {len(videos)} videos in {in_dir}")
    tqdm.write(f"[INFO] Output directory: {out_dir}")
    tqdm.write(f"[INFO] Settings: clusters={args.clusters}, sample_every={args.sample_every}, "
               f"batch_size={args.batch_size}, resize_width={args.resize_width}, random_state={args.random_state}")

    for vp in tqdm(videos, desc="Videos", unit="video"):
        process_video(
            video_path=vp,
            output_dir=out_dir,
            clusters=args.clusters,
            sample_every=args.sample_every,
            random_state=args.random_state,
            batch_size=args.batch_size,
            resize_width=args.resize_width,
        )

    tqdm.write("[DONE] All videos processed.")

if __name__ == "__main__":
    main()
