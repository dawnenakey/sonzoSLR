import depthai as dai
import cv2
import time
import os
from datetime import datetime

class OAKCameraHandler:
    def __init__(self, output_dir="temp_recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.pipeline = None
        self.device = None
        self.recording = False
        self.video_writer = None
        
    def initialize(self):
        """Initialize the OAK camera pipeline"""
        try:
            # Create pipeline
            self.pipeline = dai.Pipeline()
            
            # Define sources and outputs
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            
            xout_rgb.setStreamName("rgb")
            
            # Properties
            cam_rgb.setPreviewSize(640, 480)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            
            # Linking
            cam_rgb.preview.link(xout_rgb.input)
            
            # Connect to device
            self.device = dai.Device(self.pipeline)
            print("OAK camera initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing OAK camera: {str(e)}")
            return False
    
    def start_recording(self):
        """Start recording video"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
            self.recording = True
            print(f"Started recording to {filename}")
    
    def stop_recording(self):
        """Stop recording video"""
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print("Recording stopped")
    
    def capture_frame(self):
        """Capture a single frame from the camera"""
        if not self.device:
            return None
            
        q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        in_rgb = q_rgb.get()
        
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            if self.recording and self.video_writer:
                self.video_writer.write(frame)
            return frame
        return None
    
    def close(self):
        """Close the camera connection"""
        self.stop_recording()
        if self.device:
            self.device.close()
            self.device = None
        print("Camera connection closed")

# Example usage
if __name__ == "__main__":
    camera = OAKCameraHandler()
    if camera.initialize():
        try:
            while True:
                frame = camera.capture_frame()
                if frame is not None:
                    cv2.imshow("OAK Camera", frame)
                    
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not camera.recording:
                        camera.start_recording()
                    else:
                        camera.stop_recording()
                        
        finally:
            camera.close()
            cv2.destroyAllWindows() 