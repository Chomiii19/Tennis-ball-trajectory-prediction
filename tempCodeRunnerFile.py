from ultralytics import YOLO
import cv2
import numpy as np

# Load trained YOLO model
model = YOLO("models/last.pt")

# Initialize video capture
video_path = "input_videos/input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

# Initialize VideoWriter to save the output video
output_video_path = "output_video.avi"  # Output video file name
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for the video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Lists to store actual trajectory points
actual_trajectory = []

# Number of future frames to predict
prediction_length = 10

# Frame ID for saving output
frame_id = 0

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is left

    # Predict with YOLO
    result = model.predict(frame, conf=0.2)

    # Process detections in the current frame
    for detection in result[0].boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = detection.xyxy[0]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Filter based on size and shape (assume small round shape for the ball)
        box_width, box_height = x2 - x1, y2 - y1
        if min(box_width, box_height) < 10 or max(box_width, box_height) > 50:
            continue  # Skip blobs that are unlikely to be the ball

        # Add current position to actual trajectory
        actual_trajectory.append((center_x, center_y))

        # Draw bounding box and actual trajectory point (blue)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(
            frame, (center_x, center_y), 5, (255, 0, 0), -1
        )  # Blue for actual trajectory

    # Draw actual trajectory line (blue)
    for i in range(1, len(actual_trajectory)):
        cv2.line(frame, actual_trajectory[i - 1], actual_trajectory[i], (255, 0, 0), 2)

    # Predict future trajectory based on recent points (tracklets and smoothing)
    if len(actual_trajectory) >= 2:
        # Get recent points for smoothing and prediction
        recent_points = np.array(actual_trajectory[-5:])  # Use last 5 points
        x_coords = recent_points[:, 0]
        y_coords = recent_points[:, 1]

        # Fit a polynomial (1st degree for linear, or 2nd degree for slight curve)
        poly_fit = np.polyfit(x_coords, y_coords, 2)  # Change degree for complexity
        poly_func = np.poly1d(poly_fit)

        # Predict future positions based on fit
        for i in range(1, prediction_length + 1):
            # Calculate future x position and corresponding y
            future_x = x_coords[-1] + i * (x_coords[-1] - x_coords[-2])
            future_y = int(poly_func(future_x))

            # Keep future coordinates within frame
            future_x = max(0, min(width - 1, int(future_x)))
            future_y = max(0, min(height - 1, int(future_y)))

            # Draw the predicted trajectory (dashed red line)
            cv2.circle(frame, (future_x, future_y), 3, (0, 0, 255), -1)

    # Write the processed frame to the output video
    out.write(frame)
    frame_id += 1  # Increment frame_id for the next frame

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_video_path}")
