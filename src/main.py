"""
Camera Service API for SpokHand SLR

This service provides camera functionality and sign language recognition endpoints.
"""

from flask import Flask, jsonify, request, Response
import cv2
import numpy as np
import time
import threading
import base64
from camera.brio_asl import BRIOASLCamera
from camera.oak_camera_handler import OAKCameraHandler
import mediapipe as mp
from utils.sign_vocabulary import get_sign_categories, get_signs_in_category
from sign_spotting_service import AdvancedSignSpottingService

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global camera state
camera = None
camera_active = False
recording = False
current_frame = None
frame_lock = threading.Lock()

# Initialize sign spotting service
sign_spotting_service = AdvancedSignSpottingService()

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'camera_active': camera_active,
        'recording': recording
    })

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera capture"""
    global camera, camera_active
    
    try:
        data = request.get_json() or {}
        camera_type = data.get('camera_type', 'Logitech BRIO')
        
        if camera_active:
            return jsonify({'error': 'Camera already active'}), 400
            
        if camera_type == "Logitech BRIO":
            camera = BRIOASLCamera()
        else:  # OAK-D
            camera = OAKCameraHandler()
            
        if camera:
            if isinstance(camera, BRIOASLCamera):
                success = camera.start_capture()
            else:
                success = camera.initialize()
                
            if success:
                camera_active = True
                # Start frame capture thread
                threading.Thread(target=capture_frames, daemon=True).start()
                return jsonify({'message': f'Started {camera_type} camera'})
            else:
                return jsonify({'error': f'Failed to start {camera_type} camera'}), 500
        else:
            return jsonify({'error': 'Failed to initialize camera'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Error initializing camera: {str(e)}'}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera capture"""
    global camera, camera_active, recording
    
    try:
        if camera_active and camera:
            if isinstance(camera, BRIOASLCamera):
                camera.stop_capture()
            else:
                camera.close()
            camera = None
            camera_active = False
            recording = False
            return jsonify({'message': 'Camera stopped'})
        else:
            return jsonify({'error': 'No camera active'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error stopping camera: {str(e)}'}), 500

@app.route('/api/camera/frame')
def get_frame():
    """Get current camera frame as base64 image"""
    global current_frame
    
    if not camera_active or current_frame is None:
        return jsonify({'error': 'No camera frame available'}), 404
        
    try:
        with frame_lock:
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', current_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'frame': frame_base64})
    except Exception as e:
        return jsonify({'error': f'Error encoding frame: {str(e)}'}), 500

@app.route('/api/signs/categories')
def get_categories():
    """Get available sign categories"""
    try:
        categories = get_sign_categories()
        return jsonify({'categories': categories})
    except Exception as e:
        return jsonify({'error': f'Error getting categories: {str(e)}'}), 500

@app.route('/api/signs/category/<category>')
def get_signs_in_category_api(category):
    """Get signs in a specific category"""
    try:
        signs = get_signs_in_category(category)
        return jsonify({'signs': signs})
    except Exception as e:
        return jsonify({'error': f'Error getting signs: {str(e)}'}), 500

@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    """Start recording a sign"""
    global recording
    
    try:
        data = request.get_json()
        if not data or 'sign' not in data:
            return jsonify({'error': 'Sign parameter required'}), 400
            
        if not camera_active:
            return jsonify({'error': 'Camera not active'}), 400
            
        if recording:
            return jsonify({'error': 'Already recording'}), 400
            
        sign = data['sign']
        recording = True
        
        # Start recording thread
        threading.Thread(target=record_sign, args=(sign,), daemon=True).start()
        
        return jsonify({'message': f'Recording sign: {sign}'})
        
    except Exception as e:
        return jsonify({'error': f'Error starting recording: {str(e)}'}), 500

@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    """Stop recording"""
    global recording
    
    try:
        if not recording:
            return jsonify({'error': 'Not recording'}), 400
            
        recording = False
        return jsonify({'message': 'Recording stopped'})
        
    except Exception as e:
        return jsonify({'error': f'Error stopping recording: {str(e)}'}), 500

def capture_frames():
    """Background thread to capture camera frames"""
    global current_frame, camera_active
    
    while camera_active and camera:
        try:
            if isinstance(camera, BRIOASLCamera):
                frame = camera.get_frame()
            else:
                frame = camera.capture_frame()
                
            if frame is not None:
                with frame_lock:
                    current_frame = frame.copy()
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"Error capturing frame: {e}")
            time.sleep(0.1)

def record_sign(sign):
    """Background thread to record a sign"""
    global recording
    
    frames = []
    start_time = time.time()
    
    while recording and camera_active:
        try:
            if current_frame is not None:
                with frame_lock:
                    frames.append(current_frame.copy())
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Error recording frame: {e}")
            time.sleep(0.033)
    
    if frames:
        # Process recorded frames with sign spotting service
        try:
            # Save frames to temporary video file
            temp_path = f"/tmp/recorded_sign_{int(time.time())}.mp4"
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 30.0, (width, height))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
            # Process with sign spotting service
            segments = sign_spotting_service.process_video(temp_path)
            
            # Clean up
            import os
            os.remove(temp_path)
            
            print(f"Recorded {len(frames)} frames for sign '{sign}'")
            print(f"Detected {len(segments)} sign segments")
            
        except Exception as e:
            print(f"Error processing recorded frames: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 