"""
═══════════════════════════════════════════════════════════════════
  REAL-TIME VEHICLE SPEED TRACKER & VIOLATION DETECTION SYSTEM
  ┌─────────────────────────────────────────────────────────────┐
  │  YOLOv8 + DeepSORT | HUD Overlay | Sound Alert | CSV Log   │
  └─────────────────────────────────────────────────────────────┘
  Sources  : Webcam | IP/CCTV (RTSP) | Video File
  Features : Live stats dashboard, violation snapshots, alerts
═══════════════════════════════════════════════════════════════════
"""

import cv2
import csv
import time
import threading
import os
import argparse
import numpy as np
from datetime import datetime
from collections import deque, defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ─────────────────────────────────────────────
#  Try to import sound library (optional)
# ─────────────────────────────────────────────
try:
    import winsound
    SOUND_BACKEND = "winsound"
except ImportError:
    try:
        import subprocess
        SOUND_BACKEND = "beep"
    except Exception:
        SOUND_BACKEND = None

# ══════════════════════════════════════════
#  CONFIGURATION  — tweak these as needed
# ══════════════════════════════════════════
CONFIG = {
    # Speed limit in km/h
    "speed_limit": 30,

    # Meters per pixel calibration (adjust per scene/camera)
    "meters_per_pixel": 0.05,

    # DeepSORT max age
    "max_age": 30,

    # Vehicle class IDs (COCO): car=2, moto=3, bus=5, truck=7
    "vehicle_classes": {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"},

    # Output folder for violation snapshots
    "snapshot_dir": "violation_snapshots",

    # CSV log file
    "csv_log": "violation_log.csv",

    # How many speed samples to smooth per vehicle
    "speed_smoothing": 5,

    # Minimum confidence for detection
    "min_confidence": 0.4,

    # Sound alert cooldown (seconds) to avoid repeated beeps
    "sound_cooldown": 3,
}

# ══════════════════════════════════════════
#  HUD STYLE CONSTANTS
# ══════════════════════════════════════════
CLR = {
    "green":   (57, 255, 20),
    "red":     (0, 0, 255),
    "yellow":  (0, 220, 255),
    "cyan":    (255, 230, 0),
    "white":   (240, 240, 240),
    "black":   (0, 0, 0),
    "panel":   (10, 12, 20),
    "ok_box":  (20, 80, 20),
    "vio_box": (60, 0, 0),
    "overlay": (5, 8, 16),
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ══════════════════════════════════════════
#  UTILITY HELPERS
# ══════════════════════════════════════════

def play_alert():
    """Non-blocking beep on violation."""
    def _beep():
        if SOUND_BACKEND == "winsound":
            winsound.Beep(1000, 300)
        elif SOUND_BACKEND == "beep":
            subprocess.call(["beep", "-f", "1000", "-l", "300"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=_beep, daemon=True).start()


def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1, alpha=0.7):
    """Draw a semi-transparent rounded rectangle on the frame."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text_shadow(img, text, pos, font, scale, color, thickness=1):
    """Text with drop shadow for readability."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, scale, CLR["black"], thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


# ══════════════════════════════════════════
#  HUD OVERLAY RENDERER
# ══════════════════════════════════════════

class HUDRenderer:
    """Renders the live dashboard panel onto each frame."""

    def __init__(self, width, height, fps, source_label):
        self.width      = width
        self.height     = height
        self.fps        = fps
        self.source     = source_label
        self.start_time = time.time()

    def render(self, frame, stats: dict):
        """Draw full HUD. stats keys: total, violations, avg_speed, max_speed, active"""
        self._draw_top_bar(frame)
        self._draw_side_panel(frame, stats)
        self._draw_speed_limit_badge(frame)
        return frame

    def _draw_top_bar(self, frame):
        elapsed   = int(time.time() - self.start_time)
        ts        = datetime.now().strftime("%H:%M:%S")
        date_str  = datetime.now().strftime("%d %b %Y")
        bar_h     = 36

        draw_rounded_rect(frame, (0, 0), (self.width, bar_h),
                          CLR["overlay"], radius=0, alpha=0.82)

        # Left: source
        put_text_shadow(frame, f"⏺  {self.source}", (12, 24),
                        FONT_BOLD, 0.55, CLR["cyan"], 1)

        # Center: title
        title = "VEHICLE SPEED MONITOR"
        tw, _ = cv2.getTextSize(title, FONT_BOLD, 0.6, 1)[0], 0
        cx    = (self.width - tw[0]) // 2
        put_text_shadow(frame, title, (cx, 24), FONT_BOLD, 0.6, CLR["white"], 1)

        # Right: time + elapsed
        right_text = f"{date_str}  {ts}  |  {elapsed//60:02d}:{elapsed%60:02d}"
        rw = cv2.getTextSize(right_text, FONT, 0.48, 1)[0][0]
        put_text_shadow(frame, right_text, (self.width - rw - 12, 24),
                        FONT, 0.48, CLR["yellow"], 1)

    def _draw_side_panel(self, frame, stats):
        pw, ph = 200, 210
        px = self.width - pw - 10
        py = 46

        draw_rounded_rect(frame, (px, py), (px + pw, py + ph),
                          CLR["overlay"], radius=10, alpha=0.80)

        # Title
        put_text_shadow(frame, "LIVE STATS", (px + 12, py + 22),
                        FONT_BOLD, 0.55, CLR["cyan"], 1)
        cv2.line(frame, (px + 8, py + 30), (px + pw - 8, py + 30), CLR["cyan"], 1)

        rows = [
            ("ACTIVE",      str(stats.get("active", 0)),      CLR["green"]),
            ("TOTAL",       str(stats.get("total", 0)),        CLR["white"]),
            ("VIOLATIONS",  str(stats.get("violations", 0)),   CLR["red"]),
            ("AVG SPEED",   f"{stats.get('avg_speed', 0):.1f} km/h", CLR["yellow"]),
            ("MAX SPEED",   f"{stats.get('max_speed', 0):.1f} km/h", CLR["yellow"]),
        ]

        for i, (label, val, color) in enumerate(rows):
            y = py + 52 + i * 32
            put_text_shadow(frame, label, (px + 12, y), FONT, 0.42, CLR["white"], 1)
            vw = cv2.getTextSize(val, FONT_BOLD, 0.52, 1)[0][0]
            put_text_shadow(frame, val, (px + pw - vw - 10, y), FONT_BOLD, 0.52, color, 1)

    def _draw_speed_limit_badge(self, frame):
        bx, by = 10, 46
        draw_rounded_rect(frame, (bx, by), (bx + 90, by + 60),
                          (0, 0, 180), radius=8, alpha=0.85)
        cv2.circle(frame, (bx + 45, by + 30), 26, (255, 255, 255), 2)
        put_text_shadow(frame, "LIMIT", (bx + 18, by + 20), FONT, 0.38, CLR["white"], 1)
        limit_str = str(CONFIG["speed_limit"])
        lw = cv2.getTextSize(limit_str, FONT_BOLD, 0.8, 2)[0][0]
        put_text_shadow(frame, limit_str, (bx + 45 - lw // 2, by + 50),
                        FONT_BOLD, 0.8, CLR["white"], 2)


# ══════════════════════════════════════════
#  VIOLATION LOGGER
# ══════════════════════════════════════════

class ViolationLogger:
    def __init__(self):
        os.makedirs(CONFIG["snapshot_dir"], exist_ok=True)
        self.csv_path     = CONFIG["csv_log"]
        self._init_csv()
        self.logged       = set()
        self.last_sound   = defaultdict(float)
        self.total_count  = 0

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "track_id", "class",
                                 "speed_kmh", "snapshot_path"])

    def log(self, frame, track_id, cls_name, speed, frame_idx):
        """Log a violation: CSV row + snapshot."""
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_name = f"{CONFIG['snapshot_dir']}/violation_id{track_id}_{ts}.jpg"

        # Save snapshot
        cv2.imwrite(snap_name, frame)

        # Append CSV row
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             track_id, cls_name, f"{speed:.1f}", snap_name])

        self.total_count += 1
        print(f"  ⚠  VIOLATION — ID:{track_id} ({cls_name})  "
              f"{speed:.1f} km/h  → {snap_name}")

        # Sound alert with cooldown
        now = time.time()
        if now - self.last_sound[track_id] > CONFIG["sound_cooldown"]:
            play_alert()
            self.last_sound[track_id] = now


# ══════════════════════════════════════════
#  MAIN TRACKER CLASS
# ══════════════════════════════════════════

class VehicleSpeedTracker:

    def __init__(self, source):
        self.source_label = self._resolve_source(source)
        self.cap          = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30

        self.model   = YOLO("yolov8n.pt")
        self.tracker = DeepSort(max_age=CONFIG["max_age"])
        self.logger  = ViolationLogger()
        self.hud     = HUDRenderer(self.width, self.height,
                                   self.fps, self.source_label)

        # Per-vehicle state
        self.positions = {}
        self.speed_buf = defaultdict(lambda: deque(maxlen=CONFIG["speed_smoothing"]))
        self.seen_ids  = set()
        self.violations = set()

        # Global stats
        self.all_speeds      = []
        self.frame_idx       = 0

        # Output writer
        self.out = cv2.VideoWriter(
            "output_tracked.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )

    @staticmethod
    def _resolve_source(source):
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            return f"Webcam [{source}]"
        if isinstance(source, str) and source.lower().startswith("rtsp"):
            return "IP/CCTV Camera"
        return f"File: {os.path.basename(str(source))}"

    # ─── Main loop ───────────────────────────────

    def run(self):
        print("\n" + "═" * 60)
        print("  VEHICLE TRACKER STARTED")
        print(f"  Source  : {self.source_label}")
        print(f"  Limit   : {CONFIG['speed_limit']} km/h")
        print(f"  Logs    : {CONFIG['csv_log']}")
        print(f"  Snaps   : {CONFIG['snapshot_dir']}/")
        print("  Press Q to quit")
        print("═" * 60 + "\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_idx += 1
            detections = self._detect(frame)
            tracks     = self.tracker.update_tracks(detections, frame=frame)
            active_ids = self._draw_tracks(frame, tracks)
            stats      = self._compute_stats(active_ids)
            self.hud.render(frame, stats)

            self.out.write(frame)
            cv2.imshow("Vehicle Speed Monitor — Press Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._shutdown()

    # ─── Detection ───────────────────────────────

    def _detect(self, frame):
        results    = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id in CONFIG["vehicle_classes"] and conf >= CONFIG["min_confidence"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2, y2], conf, cls_id))
        return detections

    # ─── Drawing & speed estimation ──────────────

    def _draw_tracks(self, frame, tracks):
        active_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid  = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx, cy     = (l + r) // 2, (t + b) // 2
            center     = (cx, cy)

            cls_id   = int(track.det_class) if track.det_class is not None else 2
            cls_name = CONFIG["vehicle_classes"].get(cls_id, "Vehicle")

            self.seen_ids.add(tid)
            active_ids.add(tid)

            # ── Speed calculation ──
            speed = 0.0
            if tid in self.positions:
                px, py = self.positions[tid]
                pixel_dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                speed_raw  = (pixel_dist * CONFIG["meters_per_pixel"] /
                              (1 / self.fps)) * 3.6
                self.speed_buf[tid].append(speed_raw)
                speed = sum(self.speed_buf[tid]) / len(self.speed_buf[tid])
                self.all_speeds.append(speed)

            self.positions[tid] = center
            is_violation = speed > CONFIG["speed_limit"] and speed > 0

            # ── Box color ──
            box_color = CLR["red"] if is_violation else CLR["green"]
            cv2.rectangle(frame, (l, t), (r, b), box_color, 2)

            # ── Label background ──
            label_bg = CLR["vio_box"] if is_violation else CLR["ok_box"]
            draw_rounded_rect(frame, (l, t - 38), (r, t), label_bg,
                              radius=5, alpha=0.75)

            # ── ID + class ──
            put_text_shadow(frame, f"ID:{tid}  {cls_name}",
                            (l + 4, t - 22), FONT, 0.45, CLR["white"], 1)

            # ── Speed ──
            spd_color = CLR["red"] if is_violation else CLR["yellow"]
            put_text_shadow(frame, f"{speed:.1f} km/h",
                            (l + 4, t - 6), FONT_BOLD, 0.48, spd_color, 1)

            # ── Violation flash banner ──
            if is_violation:
                bw = r - l
                draw_rounded_rect(frame, (l, b + 2), (l + bw, b + 24),
                                  CLR["vio_box"], radius=4, alpha=0.85)
                put_text_shadow(frame, "⚠ SPEEDING", (l + 4, b + 18),
                                FONT_BOLD, 0.48, CLR["red"], 1)

                # Log only once per track (or log throttled)
                log_key = (tid, int(speed // 5))  # re-log if speed band changes
                if log_key not in self.violations:
                    self.violations.add(log_key)
                    self.logger.log(frame, tid, cls_name, speed, self.frame_idx)

        return active_ids

    # ─── Stats ───────────────────────────────────

    def _compute_stats(self, active_ids):
        recent  = [s for buf in self.speed_buf.values() for s in buf]
        avg_spd = sum(recent) / len(recent) if recent else 0.0
        max_spd = max(self.all_speeds[-200:]) if self.all_speeds else 0.0
        return {
            "active":     len(active_ids),
            "total":      len(self.seen_ids),
            "violations": self.logger.total_count,
            "avg_speed":  avg_spd,
            "max_speed":  max_spd,
        }

    # ─── Shutdown & summary ──────────────────────

    def _shutdown(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

        total_v   = len(self.seen_ids)
        vio_count = self.logger.total_count
        avg_spd   = (sum(self.all_speeds) / len(self.all_speeds)
                     if self.all_speeds else 0.0)
        max_spd   = max(self.all_speeds) if self.all_speeds else 0.0

        print("\n" + "═" * 60)
        print("  SESSION SUMMARY")
        print("═" * 60)
        print(f"  Total vehicles tracked : {total_v}")
        print(f"  Violations detected    : {vio_count}")
        print(f"  Average speed          : {avg_spd:.1f} km/h")
        print(f"  Peak speed             : {max_spd:.1f} km/h")
        print(f"  Violation log          : {CONFIG['csv_log']}")
        print(f"  Snapshots saved in     : {CONFIG['snapshot_dir']}/")
        print(f"  Output video           : output_tracked.mp4")
        print("═" * 60 + "\n")


# ══════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Vehicle Speed Tracker & Violation Detector"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--webcam", type=int, default=None,
        metavar="INDEX",
        help="Use webcam by device index (e.g. --webcam 0)"
    )
    group.add_argument(
        "--rtsp", type=str, default=None,
        metavar="URL",
        help="RTSP stream URL  (e.g. --rtsp rtsp://192.168.1.1/stream)"
    )
    group.add_argument(
        "--file", type=str, default=None,
        metavar="PATH",
        help="Video file path  (e.g. --file video.mp4)"
    )
    parser.add_argument(
        "--limit", type=int, default=CONFIG["speed_limit"],
        help=f"Speed limit in km/h (default: {CONFIG['speed_limit']})"
    )
    parser.add_argument(
        "--mpp", type=float, default=CONFIG["meters_per_pixel"],
        help=f"Metres per pixel calibration (default: {CONFIG['meters_per_pixel']})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides
    CONFIG["speed_limit"]      = args.limit
    CONFIG["meters_per_pixel"] = args.mpp

    # Resolve source
    if args.rtsp:
        source = args.rtsp
    elif args.file:
        source = args.file
    elif args.webcam is not None:
        source = args.webcam
    else:
        # Interactive fallback
        print("\n┌─ Select input source ─────────────────────┐")
        print("│  1. Webcam (default camera)               │")
        print("│  2. IP / CCTV  (RTSP URL)                 │")
        print("│  3. Video file                            │")
        print("└───────────────────────────────────────────┘")
        choice = input("  Enter choice [1/2/3]: ").strip()

        if choice == "1":
            source = 0
        elif choice == "2":
            source = input("  Enter RTSP URL: ").strip()
        else:
            source = input("  Enter video file path: ").strip()

    tracker = VehicleSpeedTracker(source)
    tracker.run()


if __name__ == "__main__":
    main()
