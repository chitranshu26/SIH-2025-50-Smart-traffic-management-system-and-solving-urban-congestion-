# yolo_utils.py
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Tuple, Optional


def load_yolo(model_path: str) -> YOLO:
    return YOLO(model_path)


def count_batch(
    model: YOLO,
    frames: List[np.ndarray],
    conf: float = 0.35,
    imgsz: int = 640,
    vehicle_classes: Optional[set[int]] = None,
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Run YOLO on a list of frames and return (counts, annotated_frames).
    If vehicle_classes is provided, keeps only those class ids.
    """
    annotated = []
    counts = []

    # Run batched inference
    results = model(frames, conf=conf, imgsz=imgsz, verbose=False)

    for frame, r in zip(frames, results):
        count = 0
        if r.boxes is not None and len(r.boxes) > 0:
            # boxes.xyxy, boxes.cls
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            if vehicle_classes is not None:
                keep = np.isin(clss, list(vehicle_classes))
            else:
                keep = np.ones_like(clss, dtype=bool)

            for (x1, y1, x2, y2), use in zip(boxes, keep):
                if not use:
                    continue
                count += 1
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(frame, f"Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        counts.append(int(count))
        annotated.append(frame)

    return counts, annotated


def make_grid(frames: List[np.ndarray], cols: int = 2) -> np.ndarray:
    """
    Make a simple grid (2x2) image from 4 frames.
    Resizes frames to match heights.
    """
    assert len(frames) == 4, "make_grid expects exactly 4 frames"
    h = min(f.shape[0] for f in frames)
    resized = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in frames]

    # pad widths to max
    w = max(f.shape[1] for f in resized)
    padded = [cv2.copyMakeBorder(f, 0, 0, 0, w - f.shape[1], cv2.BORDER_CONSTANT) for f in resized]

    top = np.hstack([padded[0], padded[1]])
    bottom = np.hstack([padded[2], padded[3]])
    grid = np.vstack([top, bottom])
    return grid
