from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import json
import uuid
from datetime import datetime
import os
from typing import Optional, Dict, Any

app = FastAPI(title="Camera Service", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global state for active sessions
active_sessions: Dict[str, Dict[str, Any]] = {}

class CameraSession(BaseModel):
    camera_type: str = "brio"  # "brio" or "oak"
    settings: Optional[Dict[str, Any]] = None

class RecordingRequest(BaseModel):
    session_id: str
    sign_name: str

@app.get("/")
async def root():
    return {"message": "Camera Service is running", "version": "1.0.0"}

@app.post("/api/camera/start")
async def start_camera_session(camera_session: CameraSession):
    """Start a new camera session"""
    session_id = str(uuid.uuid4())
    
    try:
        # Initialize camera based on type
        if camera_session.camera_type == "brio":
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 60)
        else:  # oak
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Failed to open camera")
        
        # Store session info
        active_sessions[session_id] = {
            "camera": cap,
            "camera_type": camera_session.camera_type,
            "settings": camera_session.settings or {},
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "recording": False,
            "video_writer": None
        }
        
        return {
            "session_id": session_id,
            "status": "started",
            "camera_type": camera_session.camera_type,
            "message": f"Camera session started successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start camera: {str(e)}")

@app.post("/api/camera/stop")
async def stop_camera_session(session_id: str):
    """Stop a camera session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = active_sessions[session_id]
        
        # Stop recording if active
        if session["recording"] and session["video_writer"]:
            session["video_writer"].release()
        
        # Release camera
        if session["camera"]:
            session["camera"].release()
        
        # Remove session
        del active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "status": "stopped",
            "message": "Camera session stopped successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop camera: {str(e)}")

@app.post("/api/camera/record")
async def start_recording(recording_request: RecordingRequest):
    """Start recording a sign"""
    session_id = recording_request.session_id
    sign_name = recording_request.sign_name
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    if session["recording"]:
        raise HTTPException(status_code=400, detail="Already recording")
    
    try:
        # Create output directory
        os.makedirs("recordings", exist_ok=True)
        
        # Create video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/{sign_name}_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
        
        session["recording"] = True
        session["video_writer"] = video_writer
        session["recording_filename"] = filename
        session["sign_name"] = sign_name
        
        return {
            "session_id": session_id,
            "status": "recording_started",
            "filename": filename,
            "sign_name": sign_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start recording: {str(e)}")

@app.post("/api/camera/stop-recording")
async def stop_recording(session_id: str):
    """Stop recording"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    if not session["recording"]:
        raise HTTPException(status_code=400, detail="Not recording")
    
    try:
        # Stop recording
        if session["video_writer"]:
            session["video_writer"].release()
        
        session["recording"] = False
        filename = session.get("recording_filename", "")
        
        return {
            "session_id": session_id,
            "status": "recording_stopped",
            "filename": filename,
            "message": "Recording stopped successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop recording: {str(e)}")

@app.get("/api/camera/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, session in active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "camera_type": session["camera_type"],
            "status": session["status"],
            "recording": session["recording"],
            "created_at": session["created_at"]
        })
    
    return {"sessions": sessions}

@app.websocket("/ws/camera/{session_id}")
async def camera_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time camera feed with hand tracking"""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    session = active_sessions[session_id]
    camera = session["camera"]
    
    try:
        while True:
            # Read frame from camera
            ret, frame = camera.read()
            if not ret:
                break
            
            # Resize frame for processing
            frame = cv2.resize(frame, (640, 480))
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )
            
            # Convert to base64 for WebSocket transmission
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = buffer.tobytes()
            
            # Send frame data
            await websocket.send_bytes(frame_base64)
            
            # If recording, write frame
            if session["recording"] and session["video_writer"]:
                session["video_writer"].write(frame)
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"Error in WebSocket: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 