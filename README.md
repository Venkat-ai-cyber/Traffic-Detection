# 🚦 Real-Time Vehicle Speed Tracker & Violation Detection System

> An AI-powered traffic monitoring system that detects, tracks, and logs speeding vehicles in real-time using computer vision — achieving **30+ FPS on standard CPU hardware**.

---

## 📌 Problem Statement

Manual traffic speed enforcement is resource-intensive, error-prone, and requires constant human presence. Existing automated systems depend on expensive radar hardware or GPU infrastructure. This project solves that by building a **software-only, CPU-efficient** solution using deep learning and multi-object tracking.

---

## 🎯 Features

- ✅ Real-time vehicle detection using **YOLOv8**
- ✅ Persistent multi-object tracking using **DeepSORT**
- ✅ Automatic speed estimation per vehicle
- ✅ Violation detection with **instant sound alerts**
- ✅ **Timestamped snapshots** saved for each violation
- ✅ Structured **CSV audit log** of all violations
- ✅ Runs at **30+ FPS on CPU** — no GPU required

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Object Tracking | DeepSORT |
| Computer Vision | OpenCV |
| Language | Python 3.10+ |
| Data Logging | CSV (pandas) |
| Audio Alerts | playsound / winsound |

---

## 📁 Project Structure

```
Traffic-Detection/
│
├── vehicle_tracker.py      # Main detection + tracking + speed estimation
├── violation_log.csv       # Auto-generated violation records
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Venkat-ai-cyber/Traffic-Detection.git
cd Traffic-Detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 model
The model (`yolov8n.pt`) is downloaded automatically on first run via the `ultralytics` package. No manual download needed.

---

## ▶️ Usage

```bash
python vehicle_tracker.py
```

To run on a specific video file, edit this line in `vehicle_tracker.py`:
```python
VIDEO_SOURCE = "your_video.mp4"   # or 0 for webcam
```

---

## 📊 Output

### Violation Log (`violation_log.csv`)
| Vehicle ID | Timestamp | Estimated Speed (km/h) | Snapshot Path |
|---|---|---|---|
| 3 | 2025-11-14 10:23:41 | 78.4 | snapshots/vehicle_3_10-23-41.jpg |
| 7 | 2025-11-14 10:24:05 | 91.2 | snapshots/vehicle_7_10-24-05.jpg |

---

## 🚀 How It Works

```
Video Input (live/recorded)
        │
        ▼
  YOLOv8 Detection ──── detects vehicles per frame
        │
        ▼
  DeepSORT Tracking ─── assigns persistent IDs across frames
        │
        ▼
  Speed Estimation ──── calculates speed using pixel displacement + FPS
        │
        ▼
  Violation Check ───── flags vehicles exceeding speed threshold
        │
        ▼
  Logging & Alert ───── saves snapshot + CSV entry + plays alert sound
```

---

## 📦 Requirements

Create a `requirements.txt` with:
```
ultralytics
opencv-python
deep-sort-realtime
pandas
playsound
```

Install via:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- [ ] License plate recognition using OCR (Tesseract / EasyOCR)
- [ ] Web dashboard for live monitoring (Streamlit / Flask)
- [ ] Multi-lane speed zone configuration
- [ ] GPU acceleration support (CUDA)
- [ ] Integration with traffic database / REST API

---

## 👨‍💻 Author

**Venkat Narayanan M**
AI & ML Engineering Student | Chennai Institute of Technology

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/venkat-narayanan-13b088333)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Venkat-ai-cyber)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).