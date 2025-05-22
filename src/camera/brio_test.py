import cv2
import time
import numpy as np

def test_brio_camera():
    # Initialize the BRIO camera
    cap = cv2.VideoCapture(0)
    
    # Set to 1080p resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Set frame rate to 60fps
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    print("Camera initialized with:")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Test the camera
    while True:
        ret, frame = cap.read()
        if ret:
            # Display the frame
            cv2.imshow('BRIO Test', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Could not read frame")
            break
    
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_brio_camera() 