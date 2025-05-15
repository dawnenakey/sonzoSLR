import streamlit as st
import boto3
import os
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path
import tempfile
import depthai as dai
import time
from typing import Optional, Tuple
import streamlit.web.server.server as server
from streamlit.web.server.server import Server

# AWS credentials from environment or .env file
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "spokhand-data")

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# Create Streamlit app
app = st

# Configure Streamlit for AWS deployment
server._set_websocket_headers = lambda headers: headers.update({
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
})

class OakCamera:
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.recording = False
        self.video_writer = None
        self.temp_dir = Path("temp_recordings")
        self.temp_dir.mkdir(exist_ok=True)
        
    def initialize(self) -> bool:
        try:
            # Create pipeline
            self.pipeline = dai.Pipeline()
            
            # Define sources and outputs
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            
            xout_rgb.setStreamName("rgb")
            
            # Properties
            cam_rgb.setPreviewSize(640, 480)
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            
            # Linking
            cam_rgb.preview.link(xout_rgb.input)
            
            # Connect to device
            self.device = dai.Device(self.pipeline)
            return True
        except Exception as e:
            st.error(f"Failed to initialize OAK camera: {str(e)}")
            return False
            
    def start_recording(self):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = self.temp_dir / f"recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(temp_file), fourcc, 30.0, (640, 480))
            self.recording = True
            return temp_file
            
    def stop_recording(self) -> Optional[Path]:
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                return self.temp_dir.glob("recording_*.mp4").__next__()
        return None
        
    def get_frame(self) -> Optional[np.ndarray]:
        if not self.device:
            return None
            
        q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        in_rgb = q_rgb.tryGet()
        
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            if self.recording and self.video_writer:
                self.video_writer.write(frame)
            return frame
        return None
        
    def cleanup(self):
        if self.device:
            self.device.close()
        if self.video_writer:
            self.video_writer.release()

# Main app code
def main():
    st.set_page_config(
        page_title="OAK Camera Video Upload",
        page_icon="📹",
        layout="wide"
    )

    st.title("OAK Camera Video Upload to AWS")

    # Initialize session state
    if 'camera' not in st.session_state:
        st.session_state.camera = OakCamera()
    if 'recording' not in st.session_state:
        st.session_state.recording = False

    # Create three columns
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.header("Camera Feed")
        if st.button("Initialize Camera"):
            if st.session_state.camera.initialize():
                st.success("Camera initialized successfully!")
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        
        # Recording controls
        if not st.session_state.recording:
            if st.button("Start Recording"):
                temp_file = st.session_state.camera.start_recording()
                st.session_state.recording = True
                st.success("Recording started!")
        else:
            if st.button("Stop Recording"):
                recorded_file = st.session_state.camera.stop_recording()
                st.session_state.recording = False
                if recorded_file:
                    st.success(f"Recording saved to {recorded_file}")
                    
                    # Upload to S3
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    s3_key = f"oak_videos/{timestamp}_{recorded_file.name}"
                    
                    try:
                        s3.upload_file(str(recorded_file), S3_BUCKET_NAME, s3_key)
                        st.success(f"Successfully uploaded to S3!")
                        
                        # Clean up temporary file
                        recorded_file.unlink()
                        
                        # Display S3 URL
                        s3_url = f"s3://{S3_BUCKET_NAME}/{s3_key}"
                        st.code(s3_url, language="text")
                        
                    except Exception as e:
                        st.error(f"Error uploading file: {str(e)}")
                        if recorded_file.exists():
                            recorded_file.unlink()

    with col2:
        st.header("Upload Video")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Create a temporary file to store the uploaded video
            temp_path = Path("temp_uploads")
            temp_path.mkdir(exist_ok=True)
            temp_file = temp_path / uploaded_file.name
            
            # Save the uploaded file temporarily
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Generate a unique S3 key with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"oak_videos/{timestamp}_{uploaded_file.name}"
            
            # Upload to S3
            try:
                s3.upload_file(str(temp_file), S3_BUCKET_NAME, s3_key)
                st.success(f"Successfully uploaded {uploaded_file.name} to S3!")
                
                # Clean up temporary file
                temp_file.unlink()
                
                # Display S3 URL
                s3_url = f"s3://{S3_BUCKET_NAME}/{s3_key}"
                st.code(s3_url, language="text")
                
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
                if temp_file.exists():
                    temp_file.unlink()

    with col3:
        st.header("Recent Uploads")
        try:
            # List recent uploads
            response = s3.list_objects_v2(
                Bucket=S3_BUCKET_NAME,
                Prefix="oak_videos/",
                MaxKeys=5
            )
            
            if 'Contents' in response:
                st.subheader("Last 5 uploads:")
                for obj in response['Contents']:
                    # Get file size in MB
                    size_mb = obj['Size'] / (1024 * 1024)
                    # Format last modified date
                    last_modified = obj['LastModified'].strftime("%Y-%m-%d %H:%M:%S")
                    
                    st.write(f"📁 {obj['Key'].split('/')[-1]}")
                    st.write(f"   Size: {size_mb:.2f} MB")
                    st.write(f"   Uploaded: {last_modified}")
                    
                    # Generate presigned URL for video preview
                    url = s3.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': S3_BUCKET_NAME, 'Key': obj['Key']},
                        ExpiresIn=3600
                    )
                    st.video(url)
                    st.write("---")
            else:
                st.info("No uploads found yet.")
                
        except Exception as e:
            st.error(f"Error listing uploads: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
        ### About
        This application allows you to:
        1. Record videos directly from your OAK camera
        2. Upload existing video files
        3. View and preview your uploaded videos in AWS S3
        All videos are stored in the `oak_videos/` directory of the S3 bucket.
        """)

    # Main loop for camera feed
    if st.session_state.camera.device:
        while True:
            frame = st.session_state.camera.get_frame()
            if frame is not None:
                camera_placeholder.image(frame, channels="BGR", use_column_width=True)
            time.sleep(0.01)  # Small delay to prevent high CPU usage

if __name__ == "__main__":
    main() 