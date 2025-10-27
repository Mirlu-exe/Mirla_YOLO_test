import os
import time
import cv2
from ultralytics import YOLO
from tqdm.auto import tqdm  # NEW: pretty progress bar with ETA

# === CONFIG ===
VIDEOS_DIR = os.path.join('.', 'unseen_videos')
VIDEO_NAME = '0820_8.30session_site2_1_h264.mp4'
MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'last.pt')

CONF = 0.5
IMGSZ = 640
DEVICE = 0         # 'cpu' or 0/1 for GPU
PROCESS_EVERY = 2  # process every Nth frame (reduces load)

# === INIT ===
video_path = os.path.join(VIDEOS_DIR, VIDEO_NAME)
video_path_out = f'{video_path}_out.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f'❌ Failed to open video: {video_path}')

ret, frame = cap.read()
if not ret or frame is None:
    raise RuntimeError('❌ Could not read first frame.')

H, W = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps and fps > 0 else 25
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0  # may be 0 for some streams

out = cv2.VideoWriter(
    video_path_out,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (W, H)
)

print(f'🎬 Input: {VIDEO_NAME} ({W}x{H}, {fps:.1f} fps, frames={frame_count or "unknown"})')
print(f'🧠 Loading YOLO: {MODEL_PATH}')
model = YOLO(MODEL_PATH)
print('✅ Model loaded\n')

# === MAIN LOOP with tqdm ===
frame_idx = 0
processed_frames = 0
start_time = time.time()

# total=None makes tqdm show a spinner if frame_count is unknown
total_for_bar = frame_count if frame_count > 0 else None
pbar = tqdm(total=total_for_bar, unit='f', desc='Processing', dynamic_ncols=True, smoothing=0.1)

try:
    while ret and frame is not None:
        do_infer = (frame_idx % PROCESS_EVERY == 0)

        if do_infer:
            # downscale copy for inference
            scale = IMGSZ / max(W, H)
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            results = model.predict(
                small,
                conf=CONF,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False
            )[0]

            sh, sw = small.shape[:2]
            fx, fy = W / sw, H / sh

            for x1, y1, x2, y2, score, class_id in results.boxes.data.tolist():
                if score < CONF:
                    continue
                x1, y1, x2, y2 = int(x1*fx), int(y1*fy), int(x2*fx), int(y2*fy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = results.names[int(class_id)].upper()
                cv2.putText(frame, label, (x1, max(0, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            processed_frames += 1

        out.write(frame)

        # tqdm step
        pbar.update(1)
        # show a compact live summary in the bar
        if pbar.n % 25 == 0:  # update postfix every 25 frames to keep it snappy
            infer_ratio = processed_frames / max(1, pbar.n)
            pbar.set_postfix(proc=f'{processed_frames} inf', ratio=f'{infer_ratio:.2f}')

        # gentle throttle to avoid thermal spikes
        time.sleep(0.002)

        ret, frame = cap.read()
        frame_idx += 1

except KeyboardInterrupt:
    pbar.write('⏹ Interrupted by user.')

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()

    total_time = time.time() - start_time
    print('\n✅ Done!')
    print(f'🕒 Total time: {total_time:.1f}s')
    print(f'📊 Frames written: {frame_idx} | Inference frames: {processed_frames}')
    print(f'📁 Output: {video_path_out}')
