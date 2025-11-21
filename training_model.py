from ultralytics import YOLO
import cv2

# Load a pretrained YOLO11n model
model = YOLO("yolo11m.pt")

# Train the model on the custom dataset for 100 epochs
train_results = model.train(
    data="dataset_elephants.yaml",  # Path to dataset configuration file
    epochs=150,  # Number of training epochs
    imgsz=960,
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3]) i use 0 for first available gpu
    batch=3,
    patience=50
  )
