import streamlit as st
# Setting page config as the first Streamlit command
st.set_page_config(page_title="Video Upload and Annotation Viewer", layout="wide")

"""
Streamlit app for Sign Language Recognition and Analysis.
This application provides a user interface for uploading and analyzing sign language videos
using a trained 3D CNN + LSTM model.
"""
# =====================
# Temporarily commenting out ML dependencies and code for Streamlit Cloud compatibility.
# To restore for AWS, uncomment the relevant sections.
# =====================

import pandas as pd
import tempfile

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    st.error("OpenCV (cv2) is not available. Some features may be limited.")
    CV2_AVAILABLE = False

# =====================
# Streamlit app for MP4 upload and annotation viewing
# =====================
def main():
    st.title("Video Upload and Annotation Viewer")

    st.header("Instructions")
    st.markdown("""
    - Upload an MP4 video file (from OAK camera or any other source).
    - Both 2D and 3D video files are supported (as long as they are in MP4 format).
    - You can also upload an annotation CSV file to view or analyze annotations.
    """)

    # Video upload section
    st.header("Upload and View Video")
    uploaded_video = st.file_uploader("Upload MP4 video file", type=["mp4"])
    video_type = st.selectbox("Select video type", ["Unknown", "2D", "3D"], index=0)
    if uploaded_video:
        st.success(f"Video file {uploaded_video.name} uploaded successfully!")
        st.info(f"Video type selected: {video_type}")
        # Save uploaded file temporarily and display
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        st.video(video_path)

    # Annotation file section
    st.header("View Annotations")
    uploaded_file = st.file_uploader("Upload annotation CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

    # Model & Training Visualizations
    st.header("Model & Training Visualizations")
    st.markdown("""
    The following visualizations show the model architecture (3D CNN + LSTM), confusion matrix, training progress, and other metrics from the latest training run.
    """)
    import os
    report_images = [
        ("Model Architecture (3D CNN + LSTM)", "reports/model_architecture.png"),
        ("Detailed Confusion Matrix", "reports/detailed_confusion_matrix.png"),
        ("Training Progress", "reports/training_progress.png"),
        ("Performance by Category", "reports/performance_by_category.png"),
        ("Handshape Distribution", "reports/handshape_distribution.png"),
    ]
    for title, img_path in report_images:
        if os.path.exists(img_path):
            st.subheader(title)
            st.image(img_path, use_column_width=True)
        else:
            st.info(f"{title} visualization not found.")

if __name__ == "__main__":
    main() 