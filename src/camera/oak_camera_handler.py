import depthai as dai
import cv2
import time
import os
from datetime import datetime
import numpy as np

class OAKCameraHandler:
    def __init__(self, output_dir="temp_recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.pipeline = None
        self.device = None
        self.recording = False
        self.video_writer = None
        self.is_running = False
        self.current_frame = None
        
    def initialize(self):
        """Initialize the OAK camera pipeline"""
        try:
            print("🔍 Initializing OAK camera...")
            
            # Create pipeline
            self.pipeline = dai.Pipeline()
            
            # Define sources and outputs
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            
            xout_rgb.setStreamName("rgb")
            
            # Properties - optimized for ASL recognition
            cam_rgb.setPreviewSize(640, 480)  # Preview size for display
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)  # Full resolution
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            cam_rgb.setFps(30)  # Set to 30 FPS for stability
            
            # Enable auto focus, exposure, and white balance
            cam_rgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            cam_rgb.initialControl.setAutoExposureEnable()
            cam_rgb.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            
            # Linking
            cam_rgb.preview.link(xout_rgb.input)
            
            # Connect to device
            print("🔌 Connecting to OAK device...")
            self.device = dai.Device(self.pipeline)
            
            # Get device info
            device_info = self.device.getDeviceInfo()
            print(f"✅ OAK camera initialized successfully!")
            print(f"📱 Device: {device_info.name}")
            print(f"🔧 MX ID: {device_info.getMxId()}")
            print(f"📊 State: {device_info.state}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing OAK camera: {str(e)}")
            print("💡 Make sure:")
            print("   1. OAK camera is connected via USB")
            print("   2. DepthAI library is installed: pip install depthai")
            print("   3. You have proper USB permissions")
            return False
    
    def start_capture(self):
        """Start capturing frames from the camera"""
        if not self.device:
            if not self.initialize():
                return False
                
        self.is_running = True
        print("🎬 Started OAK camera capture")
        return True
    
    def start_recording(self):
        """Start recording video"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"oak_recording_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
            self.recording = True
            print(f"📹 Started recording to {filename}")
    
    def stop_recording(self):
        """Stop recording video"""
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print("⏹️  Recording stopped")
    
    def capture_frame(self):
        """Capture a single frame from the camera"""
        if not self.device or not self.is_running:
            return None
            
        try:
            q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            in_rgb = q_rgb.get()
            
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                
                # Store current frame for live annotation
                self.current_frame = frame.copy()
                
                # Record if recording is active
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                    
                return frame
        except Exception as e:
            print(f"⚠️  Error capturing frame: {str(e)}")
            
        return None
    
    def get_frame(self):
        """Get the current frame (compatibility with BRIO interface)"""
        return self.capture_frame()
    
    def get_camera_info(self):
        """Get current camera information"""
        if not self.device:
            return None
            
        try:
            info = {
                'type': 'OAK-D',
                'resolution': '1080p',
                'fps': 30,
                'is_running': self.is_running,
                'is_recording': self.recording,
                'output_dir': self.output_dir
            }
            return info
        except:
            return None
    
    def stop_capture(self):
        """Stop capturing frames (compatibility with BRIO interface)"""
        self.is_running = False
        print("⏹️  Stopped OAK camera capture")
    
    def close(self):
        """Close the camera connection"""
        self.stop_capture()
        self.stop_recording()
        if self.device:
            self.device.close()
            self.device = None
        print("🔌 OAK camera connection closed")

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing OAK Camera Handler...")
    
    camera = OAKCameraHandler()
    if camera.initialize():
        try:
            camera.start_capture()
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                frame = camera.capture_frame()
                if frame is not None:
                    frame_count += 1
                    
                    # Display the frame
                    cv2.imshow("OAK Camera Test", frame)
                    
                    # Calculate and display FPS
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        cv2.setWindowTitle("OAK Camera Test", f"OAK Camera - FPS: {fps:.1f}")
                    
                    # Handle key presses
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        if not camera.recording:
                            camera.start_recording()
                        else:
                            camera.stop_recording()
                    elif key == ord('s'):
                        # Save a screenshot
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"oak_screenshot_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"📸 Screenshot saved: {filename}")
                        
        finally:
            camera.close()
            cv2.destroyAllWindows()
            print("✅ OAK camera test completed")
    else:
        print("❌ Failed to initialize OAK camera") 