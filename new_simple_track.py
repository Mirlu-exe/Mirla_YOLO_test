import os
from ultralytics import YOLO
from pathlib import Path

INPUT_VIDEO = '/home/nishiyamalab/Videos/Mirla_YOLO_test_videos/unseen_videos/Target/0820_8.30session_site2_1_h264.mp4'
MODEL_PATH = Path('./runs/detect/train13/weights/last.pt')

print(os.path.exists(INPUT_VIDEO))  # should print True


model = YOLO(MODEL_PATH)

results = model.track(INPUT_VIDEO,save=True,show=True,persist=True)

