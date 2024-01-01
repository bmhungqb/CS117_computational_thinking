from ultralytics import YOLO
import cv2
import  tools.dark_channel_prior as dcp
import numpy as np

def remove_noise(image):
    processed_image, alpha_map = dcp.haze_removal(image, w_size=15, a_omega=0.95, gf_w_size=200, eps=1e-6)
    return processed_image

def main():
    # Load YOLO model
    model = YOLO(pth_model)

    cap = cv2.VideoCapture(pth_source)

    # Define the codec and create a video writer object for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(pth_output, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        processed_image = remove_noise(frame)
        if success:
            # Run YOLOv8 inference on the frame
            results = model(processed_image)
            print(results)
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # Write the annotated frame to the output video
            out.write(annotated_frame)
if __name__ == "__main__":
    pth_model = r"Code/weights/best.pt"
    pth_source = r"Code/assets/video_demo.mp4"
    pth_output = r"output_video.mp4"
    main()