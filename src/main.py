import streamlit as st
import cv2
import numpy as np
import time
from camera.brio_asl import BRIOASLCamera
from camera.oak_camera_handler import OAKCameraHandler
import mediapipe as mp
from utils.sign_vocabulary import get_sign_categories, get_signs_in_category

# Set page config
st.set_page_config(
    page_title="Spokhand - Sign Language Recognition",
    page_icon="✋",
    layout="wide"
)

# Initialize AWS configuration (optional)
try:
    from aws.config import AWSConfig
    aws_config = AWSConfig()
    aws_enabled = True
except Exception as e:
    st.warning("AWS configuration not available. Some features will be disabled.")
    aws_enabled = False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Sidebar
st.sidebar.title("Spokhand Settings")

# Camera selection
camera_type = st.sidebar.selectbox(
    "Select Camera",
    ["Logitech BRIO", "OAK-D"],
    index=0
)

st.sidebar.markdown("---")

# Initialize session state
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'selected_sign' not in st.session_state:
    st.session_state.selected_sign = None

# Main content
st.title("Spokhand - Sign Language Recognition")
st.markdown("""
    Welcome to Spokhand! This application helps you learn and practice sign language using real-time hand tracking.
    """)

# Create two columns for the main content
col1, col2 = st.columns(2)

with col1:
    st.header("Live Camera Feed")
    # Placeholder for camera feed
    camera_placeholder = st.empty()
    
    # Camera controls
    start_camera = st.button("Start Camera")
    stop_camera = st.button("Stop Camera")
    
    if start_camera and not st.session_state.camera_active:
        try:
            if camera_type == "Logitech BRIO":
                st.session_state.camera = BRIOASLCamera()
            else:  # OAK-D
                st.session_state.camera = OAKCameraHandler()
                
            if st.session_state.camera:
                if isinstance(st.session_state.camera, BRIOASLCamera):
                    success = st.session_state.camera.start_capture()
                else:
                    success = st.session_state.camera.initialize()
                    
                if success:
                    st.session_state.camera_active = True
                    st.success(f"Started {camera_type} camera")
                else:
                    st.error(f"Failed to start {camera_type} camera")
        except Exception as e:
            st.error(f"Error initializing camera: {str(e)}")
            
    if stop_camera and st.session_state.camera_active and st.session_state.camera:
        try:
            if isinstance(st.session_state.camera, BRIOASLCamera):
                st.session_state.camera.stop_capture()
            else:
                st.session_state.camera.close()
            st.session_state.camera = None
            st.session_state.camera_active = False
            st.info("Camera stopped")
        except Exception as e:
            st.error(f"Error stopping camera: {str(e)}")

    # Display camera feed
    if st.session_state.camera_active and st.session_state.camera:
        while True:
            try:
                if isinstance(st.session_state.camera, BRIOASLCamera):
                    frame = st.session_state.camera.get_frame()
                else:
                    frame = st.session_state.camera.capture_frame()
                    
                if frame is not None:
                    # Process hands with MediaPipe if using BRIO
                    if isinstance(st.session_state.camera, BRIOASLCamera):
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Process the frame
                        results = mp_hands.process(rgb_frame)
                        # Draw hand landmarks
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame,
                                    hand_landmarks,
                                    mp.solutions.hands.HAND_CONNECTIONS
                                )
                    
                    # Convert to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Check for stop button (need to break the loop if camera is stopped)
                    if not st.session_state.camera_active:
                        break
                else:
                    st.error("Could not read frame from camera")
                    break
            except Exception as e:
                st.error(f"Error processing camera frame: {str(e)}")
                break

with col2:
    st.header("Sign Language Recognition")
    # Placeholder for recognized signs
    recognition_placeholder = st.empty()
    
    # Display recognized sign
    st.subheader("Current Sign")
    if st.session_state.selected_sign:
        st.markdown(f"### {st.session_state.selected_sign}")
    else:
        st.markdown("### 👋")
    
    # Confidence level
    st.subheader("Confidence")
    st.progress(0.85)

# Bottom section for data collection
st.markdown("---")
st.header("Data Collection")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Record New Sign")
    
    # Sign selection
    categories = get_sign_categories()
    selected_category = st.selectbox("Select Category", categories)
    
    if selected_category:
        signs = get_signs_in_category(selected_category)
        selected_sign = st.selectbox("Select Sign", signs)
        
        if selected_sign:
            st.session_state.selected_sign = selected_sign
    
    record_button = st.button("Start Recording" if not st.session_state.recording else "Stop Recording")
    
    if record_button:
        if not st.session_state.recording and st.session_state.selected_sign and st.session_state.camera_active:
            if isinstance(st.session_state.camera, OAKCameraHandler):
                st.session_state.camera.start_recording()
            st.session_state.recording = True
            st.success(f"Recording sign: {st.session_state.selected_sign}")
        elif st.session_state.recording:
            if isinstance(st.session_state.camera, OAKCameraHandler):
                st.session_state.camera.stop_recording()
            st.session_state.recording = False
            st.info("Recording stopped")

with col4:
    st.subheader("Upload to AWS")
    if aws_enabled:
        if st.button("Upload Data"):
            with st.spinner("Uploading to AWS..."):
                # TODO: Implement AWS upload logic
                time.sleep(2)  # Simulated upload
                st.success("Data uploaded successfully!")
    else:
        st.warning("AWS upload disabled - configuration not available")

# Cleanup MediaPipe
if not st.session_state.camera_active and mp_hands:
    try:
        mp_hands.close()
    except Exception as e:
        st.error(f"Error closing MediaPipe: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    ### About Spokhand
    Spokhand is a real-time sign language recognition system that helps bridge the communication gap
    between sign language users and non-signers.
    """) 