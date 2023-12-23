from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
from ultralytics import YOLO
from tools.dehaze import process_image
import cv2
# Load YOLO model
model = YOLO(r"Code/weights/yolov8n.pt")

# Define path to video file
source = r"Code/assets/video_demo.mp4"

# Run inference on the source
results = model(source, stream=True, save=True, device='cpu', conf=0.25)

# Initialize video clip
video_clip = VideoFileClip(source)
fps = video_clip.fps
size = video_clip.size

# Create a list to store processed frames
processed_frames = []

# Process each frame and save the video
for frame, r in zip(video_clip.iter_frames(fps=fps, dtype='uint8'), results):
    # Plot bounding boxes on the frame with customization
    im_array = r.plot(line_width=2, font_size=4)
    im = Image.fromarray(im_array[..., ::-1])  # Convert to RGB PIL image
    # frame = process_image(frame)
    frame_with_overlay = Image.blend(Image.fromarray(frame), im, alpha=0.5)

    np_frame = np.array(frame_with_overlay)

    # Append the processed frame to the list
    processed_frames.append(np_frame)

# Create an ImageSequenceClip from the processed frames
processed_clip = ImageSequenceClip(processed_frames, fps=fps)

# Save the processed video
processed_clip.write_videofile("output.mp4", codec="libx264", audio_codec="aac")
