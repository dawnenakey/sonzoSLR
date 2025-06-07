"""
Tests for camera functionality in SpokHandSLR.
"""
import pytest
import cv2
import numpy as np
from src.camera.oak_camera_handler import OAKCameraHandler
from src.camera.brio_asl import BRIOASLCamera

def test_oak_camera_initialization():
    """Test OAK camera initialization."""
    camera = OAKCameraHandler()
    assert camera is not None
    assert camera.initialize() is True
    camera.close()

def test_brio_camera_initialization():
    """Test BRIO camera initialization."""
    camera = BRIOASLCamera()
    assert camera is not None
    assert camera.initialize() is True
    camera.stop_capture()

def test_frame_capture():
    """Test frame capture functionality."""
    camera = BRIOASLCamera()
    camera.initialize()
    camera.start_capture()
    
    frame = camera.get_frame()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert len(frame.shape) == 3  # Should be a color image
    
    camera.stop_capture()

def test_camera_cleanup():
    """Test proper camera cleanup."""
    camera = BRIOASLCamera()
    camera.initialize()
    camera.start_capture()
    camera.stop_capture()
    
    # Should not raise any errors
    camera.stop_capture()

@pytest.mark.skipif(not cv2.VideoCapture(0).isOpened(), reason="No camera available")
def test_live_camera():
    """Test live camera functionality if camera is available."""
    camera = BRIOASLCamera()
    assert camera.initialize() is True
    
    camera.start_capture()
    frame = camera.get_frame()
    assert frame is not None
    
    camera.stop_capture() 