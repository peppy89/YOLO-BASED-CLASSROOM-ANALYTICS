# YOLO-BASED-CLASSROOM-ANALYTICS (YOLO + Jetson + OpenCV)

This project is a real-time classroom monitoring system that uses computer vision and YOLO-based deep learning to detect students, count occupancy, and estimate classroom engagement without storing any personal data. A single camera placed at the back of the classroom provides a live video stream, which is processed on an NVIDIA Jetson device.

The system shows the number of students, calculates a simple engagement score using spatial zones, and logs time-stamped data for later analysis. All processing is done locally to maintain privacy and efficiency.

Features

Real-time student detection using YOLO object detection

Privacy-preserving engagement estimation based on spatial heuristics

Live annotated video feed with bounding boxes and statistics

Automatic CSV logging of timestamp, student count, and engagement score

Optimized for NVIDIA Jetson edge devices

GStreamer or OpenCV camera capture support

Single-camera, low-cost setup

System Requirements

NVIDIA Jetson (Orin, Xavier, Nano, etc.) with JetPack installed

Python 3.10+

USB camera (e.g., Sandberg)

SSD or SD card with Ubuntu

YOLO11n model weights (yolo11n.pt)

Virtual environment recommended

Installation
1. Create Project Folder
mkdir ~/classroom_yolo
cd ~/classroom_yolo

2. Create Virtual Environment
sudo apt install -y python3.10-venv python3-pip-whl python3-setuptools-whl
python3 -m venv ~/yolo_env
source ~/yolo_env/bin/activate

3. Install Dependencies
pip install --upgrade pip setuptools wheel
pip install "numpy<2.0" opencv-python ultralytics
pip install https://github.com/sudoRicheek/jetson-wheels/releases/download/jp6-cu126/torch-2.8.0-cp310-cp310-linux_aarch64.whl

4. Download YOLO Model
yolo download model=yolo11n.pt


Place the yolo11n.pt file inside your project folder.

Running the System
1. Activate Environment
source ~/yolo_env/bin/activate
cd ~/classroom_yolo

2. Run the Script
python classroom_monitor.py

3. Quit Program

Press q in the video window.

How It Works
1. Camera Capture

The system opens the camera using either:

GStreamer pipeline (recommended for Jetson), or

OpenCV VideoCapture(0) as backup.

2. YOLO Detection

Each frame is passed through YOLO11 to detect students (person class only).
The total number of detected students is shown on screen.

3. Engagement Estimation

A simple spatial heuristic determines engagement:

X-axis middle 1/3 of the image

Y-axis upper 2/3 of the image

Students whose center falls in this zone are counted as “engaged”.

4. Smoothing

Engagement is averaged over the last 30 frames to avoid sudden jumps.

5. Logging

Every 10 seconds, one line is added to classroom_log.csv:

timestamp, num_students, engagement

6. Display

The system shows:

Video with bounding boxes

Student count

Engagement percentage

Engagement-zone rectangle

Project Structure
classroom_yolo/
│
├── classroom_monitor.py     # Main program
├── yolo11n.pt               # YOLO model weights
├── classroom_log.csv        # Auto-generated log file
└── README.md                # Project documentation

Future Enhancements

Head-pose estimation for more accurate engagement

YOLO pose model for keypoints and body posture

Web dashboard for analytics visualization

Multi-camera support

Seat-map occupancy detection

Better behavior detection (phone use, hand-raising, etc.)

Privacy Notice

The system does not store images, video, or facial data.
It only logs anonymous aggregate values (student count + engagement score).
All processing runs locally on the Jetson device.

Credits

YOLO models by Ultralytics

Jetson wheels by sudoRicheek

Guidance and explanation provided by ChatGPT (OpenAI)
