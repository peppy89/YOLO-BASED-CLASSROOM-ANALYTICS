# YOLO-BASED-CLASSROOM-ANALYTICS (YOLO + Jetson + OpenCV)

A real-time, privacy-friendly classroom monitoring solution built using **YOLO object detection**, **OpenCV**, and **NVIDIA Jetson** hardware.  
The system detects students, estimates engagement, and logs analytics — all without storing identities or video.

---

## ✨ Features

- 🎯 Real-time student detection using YOLO  
- 👁️ Engagement estimation using spatial zones (middle 1/3 × upper 2/3)  
- 🔒 Privacy-preserving (no facial recognition, no image storage)  
- 📈 Automatic CSV logging for analytics  
- ⚡ Optimized for NVIDIA Jetson devices  
- 🎥 Live annotated video feed  
- 📷 Works with a single USB camera  

---

## 🧰 System Requirements

- NVIDIA Jetson (Nano, Xavier, Orin)  
- Ubuntu + JetPack  
- Python 3.10+  
- USB webcam (e.g., Sandberg)  
- YOLO11 model weights (`yolo11n.pt`)  

---

## 🚀 Installation

### 1️⃣ Create Project Folder
```bash
mkdir ~/classroom_yolo
cd ~/classroom_yolo
```

### 2️⃣ Create Virtual Environment
```bash
sudo apt install -y python3.10-venv python3-pip-whl python3-setuptools-whl
python3 -m venv ~/yolo_env
source ~/yolo_env/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install "numpy<2.0" opencv-python ultralytics
pip install https://github.com/sudoRicheek/jetson-wheels/releases/download/jp6-cu126/torch-2.8.0-cp310-cp310-linux_aarch64.whl
```

### 4️⃣ Download YOLO Model
```bash
yolo download model=yolo11n.pt
```
Move yolo11n.pt to ~/classroom_yolo.

## ▶️ Running the Program
### Activate environment:
```bash
source ~/yolo_env/bin/activate
cd ~/classroom_yolo
```
### Run:
```bash
python3 classroom_monitor.py
```
### Exit:
Press q in the video window.

## 🧠 How It Works

### 1. Camera Capture
The system tries:
  - A GStreamer pipeline (Jetson optimized), then
  - Fallback to cv2.VideoCapture(0).

### 2. YOLO Detection
  - Detects only the person class
  - Counts the number of detected students
    
### 3. Engagement Estimation
A student is considered engaged if the center of their bounding box is in:
  - The middle third horizontally, and
  - The upper two-thirds vertically
This is a simple, privacy-safe heuristic.

### 4. Smoothing
A deque(maxlen=30) keeps recent engagement values and averages them,
creating a stable engagement metric.

### 5. Logging
Every LOG_INTERVAL seconds (default 10 sec), the system writes:
```bash
timestamp, num_students, engagement
```
to classroom_log.csv.
### 6. Display
Shows:
  - Bounding boxes
  - Student count
  - Engagement %
  - Engagement zone rectangle

---

## 🔮 Future Enhancements

  - 🤖 Head-pose estimation for improved engagement accuracy
  
  - 🧍 YOLO Pose model for body posture and keypoints
  
  - 🧪 Additional behaviors (phone use, hand-raising, etc.)
  
 -  🖥️ Web dashboard (Flask + Chart.js)
  
  - 🎥 Multi-camera support
  
  - 🪑 Seat-map occupancy detection
  
  - ⚙️ TensorRT optimization

---

## 🔐 Privacy Notice

  - This system:
  
  - Does not store images or videos
  
  - Does not perform facial recognition
  
  - Logs only anonymous aggregate values
  
  - Processes all data locally on the Jetson device
