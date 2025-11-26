#!/usr/bin/env python3
"""
Classroom Attention & Occupancy Monitor
- Counts students (person detections)
- Estimates a simple "engagement" score
- Logs data to CSV over time
"""

import os
import cv2
import time
import torch
from collections import deque
from datetime import datetime
from ultralytics import YOLO


# -----------------------------
# CONFIGURATION
# -----------------------------

# Path to YOLO11 model (change if you have a different one)
MODEL_PATH = "yolo11n.pt"  # e.g. "yolo11s.pt"

# Logging
LOG_FILE = "classroom_log.csv"
LOG_INTERVAL = 10  # seconds between log entries

# Camera config
USE_GSTREAMER = True  # Set False if VideoCapture(0) works for you

# GStreamer pipeline for Jetson + USB camera (/dev/video0)
GST_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! "
    "videoconvert ! appsink"
)


# -----------------------------
# CAMERA OPEN HELPER
# -----------------------------

def open_camera():
    """
    Try to open the camera using GStreamer pipeline or plain VideoCapture(0).
    Returns: cv2.VideoCapture object or None if failed.
    """
    cap = None

    if USE_GSTREAMER:
        print("[INFO] Trying GStreamer pipeline...")
        cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[INFO] Camera opened via GStreamer.")
            return cap
        else:
            print("[WARN] Failed to open camera with GStreamer pipeline.")

    print("[INFO] Trying plain VideoCapture(0)...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[INFO] Camera opened at index 0.")
        return cap

    print("[ERROR] Could not open camera using any method.")
    return None


# -----------------------------
# MAIN
# -----------------------------

def main():
    # 1. Load YOLO model
    print("[INFO] Loading YOLO model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    if torch.cuda.is_available():
        print("[INFO] CUDA available. Using GPU.")
        model.to("cuda")
    else:
        print("[INFO] CUDA not available. Using CPU.")

    # 2. Open camera
    cap = open_camera()
    if cap is None:
        return

    # 3. Engagement smoothing + logging setup
    engagement_history = deque(maxlen=30)  # average over last 30 frames
    last_log_time = time.time()

    # Create CSV with header if first time
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,num_students,engagement\n")
        print(f"[INFO] Created log file: {LOG_FILE}")
    else:
        print(f"[INFO] Appending to existing log file: {LOG_FILE}")

    print("[INFO] Press 'q' in the video window to quit.")

    # 4. Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from camera.")
            break

        h, w, _ = frame.shape

        # 4.1 Run YOLO ONLY for persons (COCO class 0)
        # Adjust conf if needed (0.25–0.5 typical)
        results = model(frame, classes=[0], conf=0.35, verbose=False)
        boxes = results[0].boxes
        num_students = len(boxes)

        # Annotated frame from YOLO
        annotated = results[0].plot()

        # 4.2 Simple engagement heuristic
        #
        # For each detected student:
        # - Compute center of bounding box (cx, cy)
        # - If center is in the middle 1/3 horizontally and upper 2/3 vertically,
        #   we consider them "engaged" (roughly facing front).
        #
        engaged_count = 0

        for box in boxes:
            # box.xyxy shape: (1, 4)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Engagement region
            in_middle_x = (w * 0.33) < cx < (w * 0.66)
            in_upper_y = cy < (h * 0.66)

            if in_middle_x and in_upper_y:
                engaged_count += 1

        if num_students > 0:
            engagement_ratio = engaged_count / num_students
        else:
            engagement_ratio = 0.0

        engagement_history.append(engagement_ratio)
        smooth_engagement = (
            sum(engagement_history) / len(engagement_history)
            if engagement_history
            else 0.0
        )

        # 4.3 Overlay info on frame (no identities, only aggregate info)
        cv2.putText(
            annotated,
            f"Students: {num_students}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            f"Engagement: {smooth_engagement * 100:.0f}%",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Optional: draw a rectangle roughly showing the "engagement zone"
        # Middle 1/3 horizontally, upper 2/3 vertically
        x_left = int(w * 0.33)
        x_right = int(w * 0.66)
        y_bottom = int(h * 0.66)
        cv2.rectangle(
            annotated,
            (x_left, 0),
            (x_right, y_bottom),
            (255, 0, 0),
            2,
        )

        cv2.imshow("Classroom Monitor", annotated)

        # 4.4 Log periodically for dashboard
        now = time.time()
        if now - last_log_time >= LOG_INTERVAL:
            timestamp = datetime.now().isoformat(timespec="seconds")
            with open(LOG_FILE, "a") as f:
                f.write(f"{timestamp},{num_students},{smooth_engagement:.3f}\n")
            print(
                f"[LOG] {timestamp} | Students: {num_students} | Engagement: {smooth_engagement * 100:.1f}%"
            )
            last_log_time = now

        # 4.5 Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
