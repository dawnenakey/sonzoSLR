from oak_camera_handler import OAKCameraHandler
import cv2
import time

def test_camera_connection():
    print("Initializing OAK camera...")
    camera = OAKCameraHandler()
    
    if not camera.initialize():
        print("Failed to initialize camera. Please check the connection.")
        return
    
    print("\nCamera initialized successfully!")
    print("Press 'r' to start/stop recording")
    print("Press 'q' to quit")
    
    try:
        while True:
            frame = camera.capture_frame()
            if frame is not None:
                cv2.imshow("OAK Camera Test", frame)
            
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

if __name__ == "__main__":
    test_camera_connection() 