from ultralytics import YOLO
import cv2
from tools.dehaze import haze_removal
import numpy as np
# Load YOLO model
model = YOLO(r"Code/weights/yolov8n.pt")

# Define path to video file
source = r"Code/assets/video_demo.mp4"

cap = cv2.VideoCapture(source)

# Define the codec and create a video writer object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    processed_image, alpha_map = haze_removal(frame, w_size=15, a_omega=0.95, gf_w_size=200, eps=1e-6)
    cv2.imshow("img", processed_image)
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Write the annotated frame to the output video
        out.write(annotated_frame)