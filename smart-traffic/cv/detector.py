# cv/detector.py
from ultralytics import YOLO
import cv2, time, os

# 🔹 Model and video paths (relative to "sih/" folder)
MODEL_PATH = r"C:\Users\chitranshu\OneDrive\newdive\OneDrive\Desktop\sih\smart-traffic\runs\detect\train4\weights\best.pt"

VIDEO_PATH = "traffic_dataset/Vehicle_Detection_Image_Dataset/samlpe_video.mp4"


class Detector:
    def __init__(self, model_path=MODEL_PATH, conf=0.35):
        """Initialize YOLO detector with confidence threshold."""
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_map = {
            0: "vehicle",
            1: "bike",
            2: "car",
            3: "bus",
            4: "truck",
            5: "auto",
            6: "cycle"
        }

    def predict(self, frame):
        """Run YOLO prediction and return bounding boxes, labels, and confidence."""
        results = self.model.predict(frame, imgsz=640, conf=self.conf, verbose=False)[0]
        detections = []
        if results.boxes is None:
            return detections

        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = [int(x) for x in box]
            label = self.class_map.get(int(cls), str(int(cls)))
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "conf": float(conf)
            })
        return detections


def demo_video(video_path):
    """Run object detection on a given video file."""
    # Normalize path (safe for Windows/Linux)
    video_path = os.path.normpath(video_path)
    print(f"🎥 Trying to open video: {video_path}")

    # Check file existence
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video at {video_path}")
        return

    det = Detector()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ End of video reached.")
            break

        t0 = time.time()
        dets = det.predict(frame)
        t1 = time.time()

        # Draw detections
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['label']}:{d['conf']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show FPS
        fps = 1 / (t1 - t0) if (t1 - t0) > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display
        cv2.imshow("YOLO Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Exiting on user request.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_video(VIDEO_PATH)   # 🔹 Always use your dataset video
