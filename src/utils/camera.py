import cv2
import numpy as np
import mediapipe as mp

class CameraHandler:
    def __init__(self):
        self.cap = None
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def start_camera(self):
        """Start the camera capture"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
        return self.cap
    
    def stop_camera(self):
        """Stop the camera capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_frame(self):
        """Get a frame from the camera with hand landmarks"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )
        
        return frame
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        self.stop_camera()
        self.hands.close() 