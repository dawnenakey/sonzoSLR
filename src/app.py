"""
Streamlit app for OAK camera streaming and data collection.
"""
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import cv2
import numpy as np
from camera.oak_camera_handler import OAKCameraHandler
import time

# Create directories for saving data
DATA_DIR = Path("data/collected")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def main():
    st.title("OAK Camera Data Collection")
    
    # Initialize session state
    if 'camera_handler' not in st.session_state:
        st.session_state.camera_handler = OAKCameraHandler()
    
    # Sidebar controls
    st.sidebar.title("Controls")
    
    if st.sidebar.button("Start Camera"):
        try:
            if st.session_state.camera_handler.connect():
                st.success("Camera connected successfully!")
            else:
                st.error("Failed to connect to camera")
        except Exception as e:
            st.error(f"Error connecting to camera: {str(e)}")
    
    if st.sidebar.button("Stop Camera"):
        try:
            st.session_state.camera_handler.stop_streaming()
            st.success("Camera stopped successfully!")
        except Exception as e:
            st.error(f"Error stopping camera: {str(e)}")
    
    # Main content area
    st.write("### Camera Feed")
    
    # Create a placeholder for the camera feed
    camera_placeholder = st.empty()
    
    # Stream the camera feed
    if st.session_state.camera_handler.device:
        try:
            for frame in st.session_state.camera_handler.start_streaming():
                # Convert frame to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Add a small delay to prevent overwhelming the system
                time.sleep(0.1)
        except Exception as e:
            st.error(f"Error streaming camera: {str(e)}")
    else:
        st.info("Click 'Start Camera' to begin streaming")
    
    # Data collection section
    st.write("### Data Collection")
    
    if st.button("Capture Frame"):
        if st.session_state.camera_handler.device:
            try:
                # Get the latest frame
                frame = next(st.session_state.camera_handler.start_streaming())
                
                # Save the frame
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = DATA_DIR / f"frame_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)
                
                st.success(f"Frame saved as {filename}")
            except Exception as e:
                st.error(f"Error capturing frame: {str(e)}")
        else:
            st.warning("Please start the camera first")

if __name__ == "__main__":
    main() 