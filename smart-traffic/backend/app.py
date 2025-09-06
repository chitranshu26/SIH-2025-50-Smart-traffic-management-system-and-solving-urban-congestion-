# backend/app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from scheduler.rule_scheduler import proportional_allocate
from ultralytics import YOLO
import cv2, threading
from typing import List

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")

# Video paths (update with your own video files)
video_paths = [
    "videos/lane1.mp4",
    "videos/lane2.mp4",
    "videos/lane3.mp4",
    "videos/lane4.mp4",
]

# Shared lane counts
lane_counts = {f"lane{i+1}": 0 for i in range(len(video_paths))}


def process_video(path, lane_key):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        # Count vehicles = number of detections
        lane_counts[lane_key] = len(results[0].boxes)
    cap.release()


@app.get("/run")
async def run():
    """Run detection on all videos once and return signal plan"""
    threads = []
    for i, path in enumerate(video_paths):
        t = threading.Thread(target=process_video, args=(path, f"lane{i+1}"))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    plan = proportional_allocate(lane_counts)
    return {"counts": lane_counts, "plan": plan}


@app.get("/health")
async def health():
    return {"ok": True}


# ---------------- WebSocket for live updates ---------------- #

active_connections: List[WebSocket] = []


async def connect_client(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)


async def disconnect_client(websocket: WebSocket):
    active_connections.remove(websocket)


async def broadcast(message: str):
    """Broadcast message to all connected clients"""
    for connection in active_connections:
        await connection.send_text(message)


@app.websocket("/ws/traffic")
async def websocket_endpoint(websocket: WebSocket):
    await connect_client(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await broadcast(f"📡 Traffic Update: {data}")
    except WebSocketDisconnect:
        await disconnect_client(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
