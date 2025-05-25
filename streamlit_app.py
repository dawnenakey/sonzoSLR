"""
Main Streamlit app entry point for OAK camera streaming.
"""
import streamlit as st
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def main():
    st.title("OAK Camera Data Collection")
    
    # System check section
    st.sidebar.title("System Check")
    
    # Check if running on Linux
    is_linux = sys.platform.startswith('linux')
    st.sidebar.write(f"Operating System: {'Linux' if is_linux else 'Non-Linux'}")
    
    if not is_linux:
        st.warning("""
        ⚠️ This application requires Linux for OAK camera operation.
        Please ensure you're running this on a Linux system with the OAK camera connected.
        """)
    
    # Main content area
    st.write("### Camera Status")
    
    if is_linux:
        try:
            # Only import OAK camera handler if on Linux
            from src.camera.oak_camera_handler import OAKCameraHandler
            
            # Initialize session state
            if 'camera_handler' not in st.session_state:
                st.session_state.camera_handler = OAKCameraHandler()
            
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
            
            # Camera feed section
            st.write("### Camera Feed")
            camera_placeholder = st.empty()
            
            if st.session_state.camera_handler.device:
                try:
                    for frame in st.session_state.camera_handler.start_streaming():
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                except Exception as e:
                    st.error(f"Error streaming camera: {str(e)}")
            else:
                st.info("Click 'Start Camera' to begin streaming")
                
        except ImportError as e:
            st.error(f"""
            Error importing OAK camera dependencies: {str(e)}
            
            Please ensure you have installed all required dependencies:
            ```
            pip install -r requirements.txt
            ```
            """)
    else:
        st.info("""
        To use the OAK camera:
        1. Connect the OAK camera to a Linux system
        2. Install the required dependencies
        3. Run this application on that system
        """)

if __name__ == "__main__":
    main() 