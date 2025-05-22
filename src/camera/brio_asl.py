import cv2
import numpy as np
import time

class BRIOASLCamera:
    def __init__(self):
        self.cap = None
        self.is_running = False
        
    def initialize(self):
        """Initialize the BRIO camera with optimal settings for ASL recognition"""
        self.cap = cv2.VideoCapture(0)
        
        # Set to 1080p resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Set frame rate to 60fps
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Set exposure to auto
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        # Set white balance to auto
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        
        return self.cap.isOpened()
        
    def start_capture(self):
        """Start capturing frames from the camera"""
        if not self.cap or not self.cap.isOpened():
            if not self.initialize():
                return False
                
        self.is_running = True
        return True
        
    def get_frame(self):
        """Get a frame from the camera"""
        if not self.is_running:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return frame
        
    def stop_capture(self):
        """Stop capturing frames"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            
    def __del__(self):
        """Clean up when the object is destroyed"""
        self.stop_capture()

def test_asl_capture():
    """Test function to demonstrate ASL capture with BRIO"""
    camera = BRIOASLCamera()
    
    if not camera.initialize():
        print("Error: Could not initialize camera")
        return
        
    print("Camera initialized with:")
    print(f"Resolution: {camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {camera.cap.get(cv2.CAP_PROP_FPS)}")
    
    camera.start_capture()
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                # Display the frame
                cv2.imshow('ASL Capture', frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Could not read frame")
                break
    finally:
        camera.stop_capture()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_asl_capture() 