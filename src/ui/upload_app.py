import streamlit as st
import boto3
import os

# AWS credentials from environment or .env file
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "spokhand-data")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

st.title("Upload OAK/DepthAI Video to S3")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    s3_key = f"uploads/{uploaded_file.name}"
    s3.upload_fileobj(uploaded_file, S3_BUCKET_NAME, s3_key)
    st.success(f"Uploaded {uploaded_file.name} to S3 bucket {S3_BUCKET_NAME}!")
