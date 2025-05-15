import pympi
import streamlit as st
import cv2
import pandas as pd
from datetime import timedelta
import os

class ELANIntegration:
    def __init__(self):
        self.eaf = None
        self.video_path = None
        
    def create_new_eaf(self, video_path):
        """Create a new ELAN file for a video"""
        self.video_path = video_path
        self.eaf = pympi.Elan.Eaf()
        
        # Add default tiers for ASL annotation
        self.eaf.add_tier('ASL_Gloss')
        self.eaf.add_tier('ASL_Translation')
        self.eaf.add_tier('ASL_Notes')
        
    def add_annotation(self, tier_name, start_time, end_time, text):
        """Add an annotation to a specific tier"""
        if self.eaf is None:
            st.error("No ELAN file created yet!")
            return False
            
        # Convert times to milliseconds
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # Add annotation
        self.eaf.add_annotation(tier_name, start_ms, end_ms, text)
        return True
        
    def save_eaf(self, output_path):
        """Save the ELAN file"""
        if self.eaf is None:
            st.error("No ELAN file to save!")
            return False
        self.eaf.to_file(output_path)
        return True
        
    def load_eaf(self, eaf_path):
        """Load an existing ELAN file"""
        try:
            self.eaf = pympi.Elan.Eaf(eaf_path)
            return True
        except Exception as e:
            st.error(f"Error loading ELAN file: {str(e)}")
            return False
            
    def get_annotations(self, tier_name=None):
        """Get all annotations or annotations for a specific tier"""
        if self.eaf is None:
            return []
            
        if tier_name:
            return self.eaf.get_annotation_data_for_tier(tier_name)
        else:
            all_annotations = {}
            for tier in self.eaf.get_tier_names():
                all_annotations[tier] = self.eaf.get_annotation_data_for_tier(tier)
            return all_annotations
            
    def get_tiers(self):
        """Get all tier names"""
        if self.eaf is None:
            return []
        return self.eaf.get_tier_names()
        
    def add_tier(self, tier_name):
        """Add a new tier"""
        if self.eaf is None:
            st.error("No ELAN file created yet!")
            return False
        self.eaf.add_tier(tier_name)
        return True 