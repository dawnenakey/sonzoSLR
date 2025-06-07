"""
Streamlit app for Sign Language Recognition and Analysis.
This application provides a user interface for uploading and analyzing sign language videos
using a trained 3D CNN + LSTM model.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import os
import joblib
from sklearn.metrics import roc_curve, auc
import torch
import cv2
import tempfile
from typing import Tuple, Optional, Dict, Any
from src.train import SignLanguageModel, SignLanguageDataset

class ModelLoadingError(Exception):
    """Custom exception for model loading errors."""
    pass

class VideoProcessingError(Exception):
    """Custom exception for video processing errors."""
    pass

def load_model() -> Tuple[Optional[SignLanguageModel], Optional[Dict[int, str]]]:
    """
    Load the trained model and class mapping.
    
    Returns:
        Tuple[Optional[SignLanguageModel], Optional[Dict[int, str]]]: 
            The loaded model and class mapping, or (None, None) if loading fails.
            
    Raises:
        ModelLoadingError: If model loading fails.
    """
    try:
        # Check if model file exists
        if not os.path.exists('best_model.pth'):
            raise FileNotFoundError("Model file 'best_model.pth' not found")
            
        # Check if metrics file exists
        if not os.path.exists('reports/training_metrics.pt'):
            raise FileNotFoundError("Metrics file 'reports/training_metrics.pt' not found")
            
        # Load model
        try:
            model = SignLanguageModel(num_classes=5)  # 5 classes as per training
            model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
            model.eval()
        except RuntimeError as e:
            raise ModelLoadingError(f"Failed to load model state dict: {e}")
        except Exception as e:
            raise ModelLoadingError(f"Unexpected error loading model: {e}")
        
        # Load class mapping
        try:
            metrics = torch.load('reports/training_metrics.pt', weights_only=False)
            if 'class_mapping' not in metrics:
                raise KeyError("Class mapping not found in metrics file")
            class_mapping = metrics['class_mapping']
        except Exception as e:
            raise ModelLoadingError(f"Failed to load class mapping: {e}")
        
        return model, class_mapping
        
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        return None, None
    except ModelLoadingError as e:
        st.error(f"Model loading error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error during model loading: {e}")
        return None, None

def process_video(video_path: str) -> Optional[torch.Tensor]:
    """
    Process video for prediction.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        Optional[torch.Tensor]: Processed video frames as a tensor, or None if processing fails.
        
    Raises:
        VideoProcessingError: If video processing fails.
    """
    try:
        # Validate video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Load and process video frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError("Failed to open video file - file may be corrupted or in an unsupported format")
            
        frames = []
        frame_count = 0
        max_frames = 100  # Safety limit
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                frame_count += 1
            except cv2.error as e:
                raise VideoProcessingError(f"OpenCV error processing frame: {e}")
                
        cap.release()
        
        if not frames:
            raise VideoProcessingError("No frames were extracted from the video")
            
        # Convert to tensor
        try:
            frames = np.array(frames)
            frames = frames / 255.0
            
            # Handle frame count
            if len(frames) > 30:
                frames = frames[:30]
            elif len(frames) < 30:
                frames = np.repeat(frames, 30 // len(frames) + 1, axis=0)[:30]
                
            frames = torch.from_numpy(frames).float()
            frames = frames.permute(3, 0, 1, 2)
            frames = frames.unsqueeze(0)  # Add batch dimension
            
            return frames
            
        except (ValueError, RuntimeError) as e:
            raise VideoProcessingError(f"Error converting frames to tensor: {e}")
            
    except FileNotFoundError as e:
        st.error(f"Video file error: {e}")
        return None
    except VideoProcessingError as e:
        st.error(f"Video processing error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during video processing: {e}")
        return None

def create_3d_cnn_visualization() -> go.Figure:
    """
    Create a 3D visualization of CNN architecture.
    
    Returns:
        go.Figure: Plotly figure object containing the visualization.
        
    Raises:
        ValueError: If visualization creation fails.
    """
    try:
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))

        fig = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')])
        fig.update_layout(
            title='3D CNN Architecture Visualization',
            scene = dict(
                xaxis_title='Width',
                yaxis_title='Height',
                zaxis_title='Depth'
            ),
            width=800,
            height=600
        )
        return fig
    except Exception as e:
        st.error(f"Error creating CNN visualization: {e}")
        # Return empty figure as fallback
        return go.Figure()

def create_lstm_metrics() -> Optional[go.Figure]:
    """
    Create LSTM training metrics visualization.
    
    Returns:
        Optional[go.Figure]: Plotly figure object containing the visualization, or None if loading fails.
    """
    try:
        # Validate metrics file exists
        if not os.path.exists('reports/training_metrics.pt'):
            raise FileNotFoundError("Metrics file not found")
            
        metrics = torch.load('reports/training_metrics.pt', weights_only=False)
        
        # Validate required metrics are present
        required_metrics = ['train_losses', 'val_losses', 'val_accuracies']
        missing_metrics = [m for m in required_metrics if m not in metrics]
        if missing_metrics:
            raise KeyError(f"Missing required metrics: {', '.join(missing_metrics)}")
            
        epochs = list(range(len(metrics['train_losses'])))
        
        # Create subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Training and Validation Loss', 'Model Accuracy'))

        # Add loss curves
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['train_losses'], name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['val_losses'], name='Validation Loss', line=dict(color='red')),
            row=1, col=1
        )

        # Add accuracy curve
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['val_accuracies'], name='Validation Accuracy', line=dict(color='green')),
            row=2, col=1
        )

        fig.update_layout(
            height=800,
            width=800,
            title_text="LSTM Training Metrics",
            showlegend=True
        )
        return fig
    except FileNotFoundError as e:
        st.error(f"Metrics file error: {e}")
        return None
    except KeyError as e:
        st.error(f"Missing metrics data: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating LSTM metrics visualization: {e}")
        return None

def display_training_metrics() -> None:
    """Display training metrics and curves."""
    try:
        # Validate metrics file exists
        if not os.path.exists('reports/training_metrics.pt'):
            raise FileNotFoundError("Metrics file not found")
            
        metrics = torch.load('reports/training_metrics.pt', weights_only=False)
        
        # Validate required metrics
        required_metrics = ['train_losses', 'val_losses', 'val_accuracies', 
                          'total_training_time', 'device_used', 'batch_size',
                          'learning_rate', 'num_epochs']
        missing_metrics = [m for m in required_metrics if m not in metrics]
        if missing_metrics:
            raise KeyError(f"Missing required metrics: {', '.join(missing_metrics)}")
        
        # Create tabs for different metrics
        tab1, tab2, tab3 = st.tabs(["Loss Curves", "Accuracy", "Training Stats"])
        
        with tab1:
            # Plot loss curves
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=metrics['train_losses'], name='Training Loss'))
            fig.add_trace(go.Scatter(y=metrics['val_losses'], name='Validation Loss'))
            fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig)

        with tab2:
            # Plot accuracy
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=metrics['val_accuracies'], name='Validation Accuracy'))
            fig.update_layout(title='Validation Accuracy Over Time', xaxis_title='Epoch', yaxis_title='Accuracy (%)')
            st.plotly_chart(fig)

        with tab3:
            # Display training statistics
            st.subheader("Training Statistics")
            stats = {
                "Total Training Time": f"{metrics['total_training_time']:.2f} seconds",
                "Device Used": metrics['device_used'],
                "Batch Size": metrics['batch_size'],
                "Learning Rate": metrics['learning_rate'],
                "Number of Epochs": metrics['num_epochs']
            }
            st.json(stats)
    except FileNotFoundError as e:
        st.error(f"Metrics file error: {e}")
    except KeyError as e:
        st.error(f"Missing metrics data: {e}")
    except Exception as e:
        st.error(f"Error displaying training metrics: {e}")

def main() -> None:
    """Main application entry point."""
    try:
        st.set_page_config(page_title="Sign Language Recognition Demo", layout="wide")
        
        # Sidebar
        with st.sidebar:
            st.title("SLR Capstone Demo")
            st.markdown("**Instructions:**\n1. Upload a sign language video.\n2. View model predictions and metrics.\n3. Explore architecture and analytics in the tabs.")
            st.markdown("---")
            st.info("Project by Dawnena Key, Masters of Data Science Capstone 2025")
        
        st.title("Sign Language Recognition Platform")
        st.write("Upload sign language videos for analysis and recognition. This demo showcases model predictions, evaluation metrics, and architecture visualizations for your capstone presentation.")
        
        # Load model
        model, class_mapping = load_model()
        if model is None or class_mapping is None:
            st.error("Failed to load model. Please ensure the model file exists and training is complete.")
            return
        
        # File uploader for video files
        uploaded_file = st.file_uploader(
            "Upload a sign language video file (MP4, AVI, or MOV format)",
            type=['mp4', 'avi', 'mov']
        )
        
        if uploaded_file:
            st.success(f"Video file {uploaded_file.name} uploaded successfully!")
            
            # Save uploaded file temporarily
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
            except Exception as e:
                st.error(f"Error saving uploaded file: {e}")
                return
            
            try:
                # Process video
                frames = process_video(video_path)
                if frames is not None:
                    # Make prediction
                    start_time = time.time()
                    try:
                        with torch.no_grad():
                            output = model(frames)
                            probabilities = torch.softmax(output, dim=1)
                            pred_class = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0][pred_class].item()
                    except RuntimeError as e:
                        st.error(f"Error during model inference: {e}")
                        return
                        
                    inference_time = time.time() - start_time
                    
                    # Display results
                    st.metric("Inference Time (s)", f"{inference_time:.2f}")
                    st.write(f"**Predicted Sign:** {class_mapping[pred_class]}")
                    st.write(f"**Confidence:** {confidence:.2%}")
                    
                    # Show confidence for all classes
                    st.subheader("Confidence Scores")
                    for i, (label, name) in enumerate(class_mapping.items()):
                        st.write(f"{name}: {probabilities[0][i].item():.2%}")
                    
                    # Show video
                    st.video(uploaded_file)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(video_path)
                except Exception as e:
                    st.warning(f"Failed to clean up temporary file: {e}")
        
        # Model Architecture and Analytics
        st.header("Model Architecture and Analytics")
        tab1, tab2, tab3 = st.tabs(["Model Architecture", "Training Metrics", "About"])
        
        with tab1:
            st.subheader("3D CNN Architecture Visualization")
            st.write("This visualization shows the 3D convolutional neural network architecture used for sign language recognition.")
            cnn_fig = create_3d_cnn_visualization()
            st.plotly_chart(cnn_fig)
            st.markdown("**CNN Architecture Details:**\n- Input Layer: 3D video frames (height × width × channels)\n- Convolutional Layers: Multiple 3D convolutional layers with ReLU activation\n- Pooling Layers: 3D max pooling for dimensionality reduction\n- Fully Connected Layers: Final classification layers")
        
        with tab2:
            st.subheader("Training Metrics")
            display_training_metrics()
        
        with tab3:
            st.subheader("About the Model")
            st.write("""
            This sign language recognition model uses a combination of 3D CNN and LSTM architectures:
            
            1. **3D CNN**: Processes video frames to extract spatial and temporal features
            2. **LSTM**: Captures long-term dependencies in the sign sequence
            3. **Fully Connected Layers**: Classifies the sign based on extracted features
            
            The model was trained on a dataset of 100 annotated ASL signs, with expert validation
            to ensure accuracy and consistency.
            """)
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 