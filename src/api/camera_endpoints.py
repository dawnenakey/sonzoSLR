"""
FastAPI endpoints for OAK camera streaming.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from src.camera.oak_camera_handler import OAKCameraHandler
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Global camera handler instance
camera_handler = None

def get_camera_handler():
    """Get or create camera handler instance."""
    global camera_handler
    if camera_handler is None:
        camera_handler = OAKCameraHandler()
    return camera_handler

@router.get("/camera/stream")
async def stream_camera():
    """Stream camera feed."""
    try:
        handler = get_camera_handler()
        
        def generate_frames():
            for frame in handler.start_streaming():
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                # Convert to bytes
                frame_bytes = buffer.tobytes()
                
                # Yield frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
        return StreamingResponse(
            generate_frames(),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
        
    except Exception as e:
        logger.error(f"Error streaming camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/camera/stop")
async def stop_camera():
    """Stop camera stream."""
    try:
        handler = get_camera_handler()
        handler.stop_streaming()
        return {"message": "Camera stream stopped"}
    except Exception as e:
        logger.error(f"Error stopping camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 