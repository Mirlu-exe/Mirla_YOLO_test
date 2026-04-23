from ultralytics import YOLO
import cv2

# Load a pretrained YOLO11 L model
model = YOLO("yolo11l.pt")

# Train the model on the custom dataset for 100 epochs
train_results = model.train(
    data="dataset_elephants.yaml",  # Path to dataset configuration file
    epochs=500,  # Number of training epochs
    imgsz=1024,
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3]) i use 0 for first available gpu
    batch=8,
    #patience=50

    #------------------------------------------------
    ##Optimizer config, learning rate config
    optimizer="AdamW",
    lr0=0.00095,
    #warmup_epochs=3,
    #cos_lr=True,
    #freeze=23,
    #seed=0,
    #pretrained=True,
    #------------------------------------------------

    #data augmentations
    augment=True,
    hsv_h= 0.005,   # Adjusts hue (color tone). Small changes simulate different lighting/color conditions
    hsv_s= 0.2,  # Adjusts saturation (color intensity). Higher = more vivid or washed-out colors
    hsv_v= 0.1,     # Adjusts brightness (value). Simulates darker/brighter environments

    degrees= 2.0,  # Randomly rotates the image between -10 to +10 degrees (small rotations are fine, large ones hurt realism for animals)
    translate= 0.05, # Shifts the image horizontally/vertically (up to 10% of image size)
    scale= 0.2,     # Zoom in/out. 0.5 = can shrink to 50% or enlarge significantly (too much zoom can distort object size consistency)
    shear= 0.2,     # Tilts the image (like slanting it). Helps with perspective variation (mild perspective change is OK, too much = unrealistic)

    flipud= 0.0,    # Vertical flip (up/down). 0.0 = disabled (usually unrealistic for animals)
    fliplr= 0.5,    # Horizontal flip (left/right). 50% chance per image

    mosaic= 0.3,    # Combines 4 images into 1 during training. Very strong augmentation, improves generalization (great for detection, but too much can hurt tracking consistency)
    mixup= 0.0,     # Blends two images together (20% probability). Helps regularization but can make training harder (helps generalization, but can confuse object boundaries)

  )
