import streamlit as st
import cv2
from pathlib import Path
import tempfile
import os
from datetime import timedelta
import pandas as pd

from ..elan.elan_integration import ELANIntegration

def format_time(seconds):
    """Format seconds into HH:MM:SS.mmm"""
    return str(timedelta(seconds=seconds))

def elan_interface():
    st.title("ASL Video Annotation with ELAN")
    
    # Initialize ELAN integration
    elan = ELANIntegration()
    
    # File upload
    uploaded_file = st.file_uploader("Upload ASL Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
            
        # Create new ELAN file
        elan.create_new_eaf(video_path)
        
        # Video player
        st.video(video_path)
        
        # Annotation interface
        st.subheader("Add Annotation")
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.number_input("Start Time (seconds)", min_value=0.0, step=0.1)
        with col2:
            end_time = st.number_input("End Time (seconds)", min_value=0.0, step=0.1)
            
        tier_name = st.selectbox("Select Tier", elan.get_tiers())
        annotation_text = st.text_area("Annotation Text")
        
        if st.button("Add Annotation"):
            if elan.add_annotation(tier_name, start_time, end_time, annotation_text):
                st.success("Annotation added successfully!")
                
        # View annotations
        st.subheader("Current Annotations")
        annotations = elan.get_annotations()
        
        for tier, tier_annotations in annotations.items():
            st.write(f"**{tier}**")
            if tier_annotations:
                df = pd.DataFrame(tier_annotations, columns=['Start', 'End', 'Text'])
                df['Start'] = df['Start'].apply(lambda x: format_time(x/1000))
                df['End'] = df['End'].apply(lambda x: format_time(x/1000))
                st.dataframe(df)
            else:
                st.write("No annotations yet")
                
        # Save ELAN file
        if st.button("Save ELAN File"):
            output_path = Path(video_path).with_suffix('.eaf')
            if elan.save_eaf(str(output_path)):
                st.success(f"ELAN file saved to {output_path}")
                
        # Clean up temporary file
        os.unlink(video_path) 