import cv2
import numpy as np
from ultralytics import YOLO
import os

# -------------------
# Settings
# -------------------
MODEL_PATH = r"smart-traffic\runs\detect\train4\weights\best.pt"

# Paths for 4 lane videos
VIDEO_PATHS = [
    r"traffic_dataset\Vehicle_Detection_Image_Dataset\lane1.mp4",
    r"traffic_dataset\Vehicle_Detection_Image_Dataset\lane1.mp4",
    r"traffic_dataset\Vehicle_Detection_Image_Dataset\lane1.mp4",
    r"traffic_dataset\Vehicle_Detection_Image_Dataset\lane1.mp4"
]

# Accept all vehicle classes (set to None)
VEHICLE_CLASSES = None  


# -------------------
# Vehicle counter (no polygon, full frame = lane)
# -------------------
def count_vehicles(frame, model):
    results = model.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
    count = 0

    if results.boxes is not None:
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = [int(x) for x in box]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if VEHICLE_CLASSES is None or int(cls) in VEHICLE_CLASSES:
                count += 1
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Show count
    cv2.putText(frame, f"Count: {count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return frame, count


# -------------------
# Main Loop
# -------------------
if __name__ == "__main__":
    # Check files
    for vp in VIDEO_PATHS:
        if not os.path.exists(vp):
            raise FileNotFoundError(f"Video not found at {vp}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    # Open 4 video captures
    caps = [cv2.VideoCapture(vp) for vp in VIDEO_PATHS]
    model = YOLO(MODEL_PATH)

    while True:
        lane_counts = []
        frames = []

        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                lane_counts.append(0)  # no frame means end
                frames.append(np.zeros((200, 200, 3), dtype=np.uint8))  # dummy frame
                continue

            frame, count = count_vehicles(frame, model)
            lane_counts.append(count)
            frames.append(frame)

            # Show each lane separately
            cv2.imshow(f"Lane {i+1}", frame)

        # Combine into state vector
        state = np.array(lane_counts)
        print("State vector:", state)  # [lane1, lane2, lane3, lane4]

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
