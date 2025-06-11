"""
Streamlit app for Sign Language Recognition and Analysis.
This application provides a user interface for uploading and analyzing sign language videos
using a trained 3D CNN + LSTM model.
"""
# =====================
# Temporarily commenting out ML dependencies and code for Streamlit Cloud compatibility.
# To restore for AWS, uncomment the relevant sections.
# =====================

import streamlit as st
import pandas as pd
import cv2
import tempfile

# =====================
# Streamlit app for MP4 upload and annotation viewing
# =====================
def main():
    st.set_page_config(page_title="Video Upload and Annotation Viewer", layout="wide")
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
    if uploaded_video:
        st.success(f"Video file {uploaded_video.name} uploaded successfully!")
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

if __name__ == "__main__":
    main() 