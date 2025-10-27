#source of this: https://youtu.be/uMzOcCNKr5A

import os
from ultralytics import YOLO
import cv2


##load yolov11 model
model_path = YOLO('yolo11s.pt')

#load custom model
#model_path = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model


#load video
video_path ='/home/nishiyamalab/PycharmProjects/Mirla_YOLO_test/unseen_videos/Target/0820_8.30session_site2_1_h264.mp4'
"""
video_path ='/home/nishiyamalab/PycharmProjects/Mirla_YOLO_test/reencoded_250819_10session_site1early_1.mp4'
"""

cap = cv2.VideoCapture(video_path)

# Get video dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get screen resolution
screen_width = 1920
screen_height = 1080

# Calculate new dimensions while maintaining aspect ratio
aspect_ratio = original_width / original_height
if (screen_width / aspect_ratio) <= screen_height:
    new_width = int(screen_width)
    new_height = int(screen_width / aspect_ratio)
else:
    new_width = int(screen_height * aspect_ratio)
    new_height = int(screen_height)

# Create a resizable window
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', new_width, new_height)

ret = True
#read frames
while ret:
    ret, frame = cap.read()

    if ret:
        #detect objects
        #track objects
        results = model.track(frame, persist = True, classes=[20])

        #plot results
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'): #this is for stopping with the q key
            break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()