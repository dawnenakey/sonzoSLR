import cv2
import numpy as np
import time
import platform
import subprocess
import sys

class BRIOASLCamera:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.system = platform.system()
        self.camera_index = None
        
    def _check_mac_camera_permissions(self):
        """Check and guide user for Mac camera permissions"""
        if self.system == "Darwin":  # macOS
            print("🔍 Checking Mac camera permissions...")
            
            # Check if we can access the camera
            test_cap = cv2.VideoCapture(0)
            if not test_cap.isOpened():
                print("❌ Camera access denied on Mac!")
                print("📱 Please enable camera permissions:")
                print("   1. Go to System Preferences > Security & Privacy > Privacy")
                print("   2. Select 'Camera' from the left sidebar")
                print("   3. Make sure your terminal/IDE is checked")
                print("   4. Restart your terminal/IDE")
                print("   5. Try running the application again")
                return False
            test_cap.release()
            print("✅ Mac camera permissions OK")
            return True
        return True
        
    def _find_brio_camera(self):
        """Find the BRIO camera index by testing available cameras"""
        print("🔍 Searching for BRIO camera...")
        
        # Test multiple camera indices
        for i in range(10):  # Test indices 0-9
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"📹 Camera {i}: {width}x{height} @ {fps}fps")
                
                # Check if this looks like a BRIO (high resolution, high FPS)
                if width >= 1920 and height >= 1080 and fps >= 30:
                    print(f"✅ Found BRIO camera at index {i}")
                    cap.release()
                    return i
                    
                cap.release()
        
        print("⚠️  BRIO camera not found, using default camera 0")
        return 0
        
    def initialize(self):
        """Initialize the BRIO camera with optimal settings for ASL recognition"""
        # Check Mac permissions first
        if not self._check_mac_camera_permissions():
            return False
            
        # Find the best camera
        self.camera_index = self._find_brio_camera()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera at index {self.camera_index}")
            return False
            
        # Set optimal settings for ASL recognition
        print("⚙️  Configuring camera settings...")
        
        # Set to 1080p resolution (fallback to 720p if 1080p fails)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Verify resolution was set
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if actual_width < 1280 or actual_height < 720:
            print("⚠️  1080p not supported, trying 720p...")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Set frame rate (try 60fps, fallback to 30fps)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Set exposure to auto
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        # Set white balance to auto
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        
        # Set focus to auto (if supported)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        print(f"✅ Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True
        
    def start_capture(self):
        """Start capturing frames from the camera"""
        if not self.cap or not self.cap.isOpened():
            if not self.initialize():
                return False
                
        self.is_running = True
        print("🎬 Started BRIO camera capture")
        return True
        
    def get_frame(self):
        """Get a frame from the camera"""
        if not self.is_running or not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️  Failed to read frame from BRIO camera")
            return None
            
        return frame
        
    def stop_capture(self):
        """Stop capturing frames"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            print("⏹️  Stopped BRIO camera capture")
            
    def get_camera_info(self):
        """Get current camera information"""
        if not self.cap:
            return None
            
        info = {
            'index': self.camera_index,
            'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'system': self.system,
            'is_running': self.is_running
        }
        return info
            
    def __del__(self):
        """Clean up when the object is destroyed"""
        self.stop_capture()

def test_asl_capture():
    """Test function to demonstrate ASL capture with BRIO"""
    print("🚀 Testing BRIO ASL Camera...")
    print(f"💻 System: {platform.system()} {platform.release()}")
    
    camera = BRIOASLCamera()
    
    if not camera.initialize():
        print("❌ Error: Could not initialize camera")
        return
        
    # Get camera info
    info = camera.get_camera_info()
    if info:
        print(f"📹 Camera Info: {info['width']}x{info['height']} @ {info['fps']}fps")
    
    camera.start_capture()
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame = camera.get_frame()
            if frame is not None:
                frame_count += 1
                
                # Display the frame
                cv2.imshow('BRIO ASL Capture', frame)
                
                # Calculate and display FPS
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    cv2.setWindowTitle('BRIO ASL Capture', f'BRIO ASL Capture - FPS: {fps:.1f}')
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("❌ Error: Could not read frame")
                break
    finally:
        camera.stop_capture()
        cv2.destroyAllWindows()
        print("✅ Test completed")

if __name__ == "__main__":
    test_asl_capture() 