import torch
import numpy as np
import cv2
import boto3
import json
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignLanguageInference:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path):
        """Load the trained model"""
        if model_path.startswith('s3://'):
            # Download from S3
            bucket, key = model_path.replace('s3://', '').split('/', 1)
            local_path = f'/tmp/{os.path.basename(key)}'
            self.s3 = boto3.client('s3')
            self.s3.download_file(bucket, key, local_path)
            model_path = local_path
            
        model = torch.load(model_path, map_location=self.device)
        return model
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame for inference"""
        # Resize
        frame = cv2.resize(frame, (224, 224))
        # Normalize
        frame = frame / 255.0
        # Convert to tensor
        frame = torch.from_numpy(frame).float()
        frame = frame.permute(2, 0, 1)  # HWC to CHW
        return frame
    
    def predict(self, frames):
        """Make prediction on a sequence of frames"""
        with torch.no_grad():
            # Preprocess frames
            frames = torch.stack([self.preprocess_frame(f) for f in frames])
            frames = frames.unsqueeze(0)  # Add batch dimension
            frames = frames.to(self.device)
            
            # Get prediction
            output = self.model(frames)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            
            return prediction.item(), probabilities[0][prediction].item()
    
    def run_inference(self, save_video=True):
        """Run real-time inference with OAK camera"""
        # Initialize video writer if needed
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4',
                fourcc,
                30.0,
                (640, 480)
            )
        
        # Buffer for frames
        frame_buffer = []
        buffer_size = 16  # Number of frames to process at once
        
        while True:
            # Get frame from OAK camera
            frame = q.get()
            frame = frame.getCvFrame()
            
            # Add to buffer
            frame_buffer.append(frame)
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
            
            # Process when buffer is full
            if len(frame_buffer) == buffer_size:
                prediction, confidence = self.predict(frame_buffer)
                
                # Log prediction
                logger.info(f'Prediction: {prediction}, Confidence: {confidence:.2f}')
                
                # Save to S3 if confidence is high
                if confidence > 0.8:
                    self._save_prediction(frame_buffer, prediction, confidence)
            
            # Display frame
            cv2.imshow('Sign Language Recognition', frame)
            
            # Save frame if needed
            if save_video:
                out.write(frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        if save_video:
            out.release()
        cv2.destroyAllWindows()
    
    def _save_prediction(self, frames, prediction, confidence):
        """Save prediction results to S3"""
        # Save video
        video_path = f'/tmp/prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Upload video to S3
        self.s3.upload_file(
            video_path,
            'spokhand-ml-dev',
            f'inference/{os.path.basename(video_path)}'
        )
        
        # Save metadata
        metadata = {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat(),
            'video_path': f'inference/{os.path.basename(video_path)}'
        }
        
        # Upload metadata to S3
        self.s3.put_object(
            Bucket='spokhand-ml-dev',
            Key=f'metadata/{os.path.basename(video_path).replace(".mp4", ".json")}',
            Body=json.dumps(metadata)
        )

def main():
    # Initialize inference
    inference = SignLanguageInference(
        model_path='s3://spokhand-ml-dev/models/best_model.pth'
    )
    
    # Run inference
    inference.run_inference()

if __name__ == '__main__':
    main() 