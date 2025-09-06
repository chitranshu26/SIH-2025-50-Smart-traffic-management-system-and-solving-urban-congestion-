# run_rl_with_yolo.py
"""
Run RL model with YOLO vehicle counts from 4 separate videos.

Files required:
- Trained RL model: models/traffic_dqn.zip
- YOLO model: smart-traffic/runs/detect/train4/weights/best.pt
- Videos: videos/lane1.mp4 ... videos/lane4.mp4

Usage:
    python run_rl_with_yolo.py
"""

import os
import time
import csv
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from stable_baselines3 import DQN

# ---------------------
# CONFIG - edit paths
# ---------------------
YOLO_MODEL_PATH = r"smart-traffic\runs\detect\train4\weights\best.pt"
RL_MODEL_PATH = r"models\traffic_dqn.zip"

VIDEO_PATHS = [
    r"videos\lane1.mp4",
    r"videos\lane2.mp4",
    r"videos\lane3.mp4",
    r"videos\lane4.mp4",
]

VEHICLE_CLASSES: Optional[set[int]] = None  # e.g. {2,3,5}, or None for all
CONF_THRESH = 0.35
IMGSZ = 640
DECISION_PERIOD_SEC = 2.5
LOG_CSV = True
CSV_PATH = "run_log.csv"

LANE_COLORS = [(0,255,0),(0,200,255),(255,200,0),(200,0,255)]
GREEN_TAG = (0,255,0)
RED_TAG = (0,0,255)


# ---------------------
# Helpers
# ---------------------
def check_files():
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")
    if not os.path.exists(RL_MODEL_PATH):
        raise FileNotFoundError(f"RL model not found: {RL_MODEL_PATH}")
    for p in VIDEO_PATHS:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Video not found: {p}")


def open_caps(paths: List[str]) -> List[cv2.VideoCapture]:
    return [cv2.VideoCapture(p) for p in paths]


def read_frames(caps: List[cv2.VideoCapture], target_h: int = 360):
    rets, frames = [], []
    for cap in caps:
        ret, frame = cap.read()
        rets.append(ret)
        if not ret:
            frames.append(np.zeros((target_h, target_h*16//9, 3), dtype=np.uint8))
        else:
            frames.append(frame)
    return rets, frames


def count_batch(model: YOLO, frames: List[np.ndarray]) -> tuple[list[int], list[np.ndarray]]:
    frames_copy = [f.copy() for f in frames]
    results = model.predict(frames_copy, conf=CONF_THRESH, imgsz=IMGSZ, verbose=False)

    counts, annotated = [], []
    for fr, res in zip(frames_copy, results):
        cnt = 0
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            try:
                xy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                xy = boxes.xyxy
                cls_ids = boxes.cls.astype(int)

            if VEHICLE_CLASSES is not None:
                keep_mask = np.isin(cls_ids, list(VEHICLE_CLASSES))
            else:
                keep_mask = np.ones(len(cls_ids), dtype=bool)

            for (box, keep) in zip(xy, keep_mask):
                if not keep:
                    continue
                x1, y1, x2, y2 = map(int, box[:4])
                cnt += 1
                cv2.rectangle(fr, (x1, y1), (x2, y2), (0,255,255), 2)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                cv2.circle(fr, (cx, cy), 4, (0,255,0), -1)

        cv2.putText(fr, f"Count: {cnt}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
        counts.append(int(cnt))
        annotated.append(fr)

    return counts, annotated


def make_grid(frames: List[np.ndarray]) -> np.ndarray:
    assert len(frames) == 4, "Need 4 frames"
    h = min(f.shape[0] for f in frames)
    resized = [cv2.resize(f, (int(f.shape[1]*(h/f.shape[0])), h)) for f in frames]
    max_w = max(f.shape[1] for f in resized)
    padded = [cv2.copyMakeBorder(f,0,0,0,max_w-f.shape[1],cv2.BORDER_CONSTANT) for f in resized]
    top = np.hstack([padded[0], padded[1]])
    bottom = np.hstack([padded[2], padded[3]])
    return np.vstack([top, bottom])


# ---------------------
# Main
# ---------------------
def main():
    check_files()
    caps = open_caps(VIDEO_PATHS)

    print("Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL_PATH)
    print("Loading RL model...")
    dqn = DQN.load(RL_MODEL_PATH)

    last_decision_t = 0.0
    current_action = 0

    csv_file, csv_writer = None, None
    if LOG_CSV:
        csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "lane_counts", "action"])

    try:
        while True:
            rets, frames = read_frames(caps)
            if not any(rets):
                print("All videos ended.")
                break

            counts, annotated = count_batch(yolo, frames)
            state = np.array(counts, dtype=np.float32)

            now = time.monotonic()
            if now - last_decision_t >= DECISION_PERIOD_SEC:
                try:
                    action, _ = dqn.predict(state, deterministic=True)
                except Exception:
                    action, _ = dqn.predict(state.reshape(1,-1), deterministic=True)
                current_action = int(action)
                last_decision_t = now

            # Annotate GREEN/RED per lane
            for i, f in enumerate(annotated):
                tag = f"Lane {i+1} {'GREEN' if i==current_action else 'RED'}"
                color = GREEN_TAG if i==current_action else RED_TAG
                cv2.putText(f, tag, (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                h, w = f.shape[:2]
                cv2.rectangle(f, (2,2), (w-2,h-2), LANE_COLORS[i], 4)

            grid = make_grid(annotated)
            hud = f"State: {list(state.astype(int))}   RL -> GREEN: Lane {current_action+1}"
            cv2.rectangle(grid, (0,0), (grid.shape[1], 44), (30,30,30), -1)
            cv2.putText(grid, hud, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            cv2.imshow("Traffic RL (4 videos)", grid)

            if csv_writer:
                csv_writer.writerow([time.time(), list(state.astype(int)), int(current_action)])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit by user.")
                break
    finally:
        for c in caps:
            c.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
