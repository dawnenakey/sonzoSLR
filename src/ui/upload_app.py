import streamlit as st

st.title("Upload Video to S3")

uploaded_file = st.file_uploader("Upload a video file for annotation or analysis")
if uploaded_file:
    st.success(f"File {uploaded_file.name} uploaded successfully!")
    # Add your upload/annotation logic here
