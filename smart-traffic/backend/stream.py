# WebSocket broadcast for dashboard
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

# Store connected clients
active_connections: List[WebSocket] = []

async def connect_client(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

async def disconnect_client(websocket: WebSocket):
    active_connections.remove(websocket)

async def broadcast(message: str):
    for connection in active_connections:
        await connection.send_text(message)

@app.websocket("/ws/traffic")
async def websocket_endpoint(websocket: WebSocket):
    await connect_client(websocket)
    try:
        while True:
            data = await websocket.receive_text()  # receive from client
            # Echo back for now (you can replace with vehicle counts, alerts, etc.)
            await broadcast(f"📡 Traffic Update: {data}")
    except WebSocketDisconnect:
        await disconnect_client(websocket)
