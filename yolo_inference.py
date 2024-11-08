from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("models/last.pt")
video_path = "input_videos/input_video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
actual_trajectory = []
prediction_length = 10
frame_id = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = model.predict(frame, conf=0.2)

    for detection in result[0].boxes:
        x1, y1, x2, y2 = detection.xyxy[0]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        box_width, box_height = x2 - x1, y2 - y1
        if min(box_width, box_height) < 10 or max(box_width, box_height) > 50:
            continue

        actual_trajectory.append((center_x, center_y))

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

    for i in range(1, len(actual_trajectory)):
        cv2.line(frame, actual_trajectory[i - 1], actual_trajectory[i], (255, 0, 0), 2)

    if len(actual_trajectory) >= 2:
        recent_points = np.array(actual_trajectory[-5:])
        x_coords = recent_points[:, 0]
        y_coords = recent_points[:, 1]

        poly_fit = np.polyfit(x_coords, y_coords, 2)
        poly_func = np.poly1d(poly_fit)

        for i in range(1, prediction_length + 1):
            future_x = x_coords[-1] + i * (x_coords[-1] - x_coords[-2])
            future_y = int(poly_func(future_x))
            future_x = max(0, min(width - 1, int(future_x)))
            future_y = max(0, min(height - 1, int(future_y)))
            cv2.circle(frame, (future_x, future_y), 3, (0, 255, 255), -1)

    out.write(frame)
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_video_path}")
