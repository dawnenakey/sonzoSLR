"""
OAK Camera Handler for Linux-based systems.
This module provides functionality to connect and stream data from the OAK camera.
"""
import depthai as dai
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OAKCameraHandler:
    def __init__(self):
        """Initialize OAK camera handler with Linux-specific configurations."""
        self.pipeline = None
        self.device = None
        self.running = False
        
    def initialize_pipeline(self):
        """Create and configure the OAK camera pipeline."""
        try:
            # Create pipeline
            self.pipeline = dai.Pipeline()
            
            # Define sources and outputs
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            
            xout_rgb.setStreamName("rgb")
            
            # Properties
            cam_rgb.setPreviewSize(300, 300)
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            
            # Linking
            cam_rgb.preview.link(xout_rgb.input)
            
            logger.info("Pipeline created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            return False
            
    def connect(self):
        """Connect to the OAK camera device."""
        try:
            if not self.pipeline:
                if not self.initialize_pipeline():
                    return False
                    
            # Connect to device
            self.device = dai.Device(self.pipeline)
            logger.info("Connected to OAK camera")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to OAK camera: {str(e)}")
            return False
            
    def start_streaming(self):
        """Start streaming from the OAK camera."""
        if not self.device:
            if not self.connect():
                return False
                
        self.running = True
        q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        while self.running:
            try:
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()
                yield frame
                
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                break
                
    def stop_streaming(self):
        """Stop the camera stream."""
        self.running = False
        if self.device:
            self.device.close()
            logger.info("Camera stream stopped")
            
    def __del__(self):
        """Cleanup when the handler is destroyed."""
        self.stop_streaming() 