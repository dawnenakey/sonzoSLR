from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import boto3
import uuid
import json
import os
from datetime import datetime
from typing import List, Optional
import cv2
import numpy as np
from pydantic import BaseModel
import mediapipe as mp
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SpokHand Data Collection Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "spokhand-data")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "spokhand-data-collection")

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SessionData(BaseModel):
    user_id: str
    session_name: str
    description: Optional[str] = None
    tags: List[str] = []

class VideoMetadata(BaseModel):
    session_id: str
    filename: str
    duration: float
    frame_count: int
    resolution: str
    hand_landmarks_detected: bool
    upload_timestamp: str

@app.get("/")
async def root():
    return {"message": "SpokHand Data Collection Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/sessions/create")
async def create_session(session_data: SessionData):
    """Create a new data collection session"""
    try:
        session_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        session_item = {
            "session_id": session_id,
            "timestamp": timestamp,
            "user_id": session_data.user_id,
            "session_name": session_data.session_name,
            "description": session_data.description,
            "tags": session_data.tags,
            "status": "active",
            "video_count": 0,
            "total_duration": 0.0
        }
        
        table.put_item(Item=session_item)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/upload-video")
async def upload_video(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload video for a specific session"""
    try:
        # Validate session exists
        response = table.get_item(Key={"session_id": session_id, "timestamp": "session"})
        if "Item" not in response:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Generate unique filename with proper folder structure
        file_extension = file.filename.split(".")[-1]
        video_id = str(uuid.uuid4())
        filename = f"videos/{session_id}/{video_id}.{file_extension}"
        
        # Upload to S3
        s3_client.upload_fileobj(
            file.file,
            S3_BUCKET,
            filename,
            ExtraArgs={"ContentType": f"video/{file_extension}"}
        )
        
        # Process video in background
        background_tasks.add_task(process_video_metadata, session_id, filename, video_id)
        
        return {
            "success": True,
            "video_id": video_id,
            "filename": filename,
            "message": "Video uploaded successfully"
        }
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video_metadata(session_id: str, filename: str, video_id: str):
    """Process video metadata and extract hand landmarks"""
    try:
        # Download video from S3 for processing
        local_path = f"/tmp/{video_id}.mp4"
        s3_client.download_file(S3_BUCKET, filename, local_path)
        
        # Extract video metadata
        cap = cv2.VideoCapture(local_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = f"{width}x{height}"
        
        # Check for hand landmarks
        hand_landmarks_detected = await detect_hand_landmarks(local_path)
        
        # Store metadata in DynamoDB
        metadata_item = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "video_id": video_id,
            "filename": filename,
            "duration": duration,
            "frame_count": frame_count,
            "resolution": resolution,
            "hand_landmarks_detected": hand_landmarks_detected,
            "processing_status": "completed"
        }
        
        table.put_item(Item=metadata_item)
        
        # Update session statistics
        update_session_stats(session_id, duration)
        
        # Clean up local file
        os.remove(local_path)
        
        logger.info(f"Video processing completed for {video_id}")
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        # Store error status
        error_item = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "video_id": video_id,
            "filename": filename,
            "processing_status": "error",
            "error_message": str(e)
        }
        table.put_item(Item=error_item)

async def detect_hand_landmarks(video_path: str) -> bool:
    """Detect hand landmarks in video using MediaPipe"""
    try:
        cap = cv2.VideoCapture(video_path)
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            frame_count = 0
            max_frames_to_check = 30  # Check first 30 frames
            
            while cap.isOpened() and frame_count < max_frames_to_check:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    cap.release()
                    return True
                
                frame_count += 1
        
        cap.release()
        return False
        
    except Exception as e:
        logger.error(f"Error detecting hand landmarks: {str(e)}")
        return False

def update_session_stats(session_id: str, duration: float):
    """Update session statistics"""
    try:
        response = table.update_item(
            Key={"session_id": session_id, "timestamp": "session"},
            UpdateExpression="SET video_count = video_count + :inc, total_duration = total_duration + :duration",
            ExpressionAttributeValues={
                ":inc": 1,
                ":duration": duration
            },
            ReturnValues="UPDATED_NEW"
        )
    except Exception as e:
        logger.error(f"Error updating session stats: {str(e)}")

@app.get("/sessions/{session_id}/videos")
async def get_session_videos(session_id: str):
    """Get all videos for a session"""
    try:
        response = table.query(
            KeyConditionExpression="session_id = :session_id",
            ExpressionAttributeValues={":session_id": session_id}
        )
        
        videos = [item for item in response.get("Items", []) if "video_id" in item]
        return {
            "success": True,
            "videos": videos,
            "count": len(videos)
        }
    except Exception as e:
        logger.error(f"Error fetching session videos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/user/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all sessions for a user"""
    try:
        response = table.query(
            IndexName="UserSessionsIndex",
            KeyConditionExpression="user_id = :user_id",
            ExpressionAttributeValues={":user_id": user_id}
        )
        
        sessions = [item for item in response.get("Items", []) if "session_name" in item]
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        logger.error(f"Error fetching user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos/{video_id}/stream")
async def stream_video(video_id: str):
    """Generate presigned URL for video streaming"""
    try:
        # Find video in DynamoDB
        response = table.scan(
            FilterExpression="video_id = :video_id",
            ExpressionAttributeValues={":video_id": video_id}
        )
        
        if not response.get("Items"):
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_item = response["Items"][0]
        filename = video_item["filename"]
        
        # Generate presigned URL
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': filename},
            ExpiresIn=3600  # 1 hour
        )
        
        return {
            "success": True,
            "stream_url": presigned_url,
            "metadata": video_item
        }
    except Exception as e:
        logger.error(f"Error generating stream URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 