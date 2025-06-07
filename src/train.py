"""
train.py

Capstone: Sign Language Recognition with WLASL

- Trains a 3D CNN + LSTM model on sign language video data.
- Plots and saves confusion matrix and loss/accuracy curves to the reports/ directory.
- Designed for reproducibility and easy grading.

How to use:
1. Ensure data/raw/ contains .mp4 videos and matching .json annotations.
2. Run: python src/train.py
3. Find plots in reports/ for your presentation.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import boto3
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignLanguageDataset(Dataset):
    """Custom Dataset for loading sign language videos and annotations."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])
        try:
            self.samples = self._load_samples()
            # Create label mapping
            unique_labels = sorted(set(label for _, label in self.samples))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            logger.info(f"Label mapping: {self.label_to_idx}")
            logger.info(f"Found {len(self.samples)} samples in {data_dir}")
            if len(self.samples) > 0:
                logger.info(f"First few samples: {self.samples[:3]}")
        except Exception as e:
            logger.error(f"Error loading samples: {e}")
            self.samples = []
            raise
        
    def _load_samples(self):
        # Load video paths and annotations
        samples = []
        logger.info(f"Searching for videos in: {self.data_dir}")
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            logger.info(f"Checking directory: {root}")
            logger.info(f"Found {len(files)} files")
            
            # Look for video files
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    # Get class label from directory name
                    class_label = os.path.basename(os.path.dirname(file_path))
                    try:
                        class_label = int(class_label)
                        samples.append((file_path, class_label))
                        logger.info(f"Found video: {file_path} with label {class_label}")
                    except ValueError:
                        logger.warning(f"Could not parse class label from directory: {class_label}")

        if not samples:
            logger.warning(f"No video files found in {self.data_dir}")
        else:
            logger.info(f"Found {len(samples)} total videos")
            logger.info(f"Unique classes: {set(label for _, label in samples)}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_video(self, video_path):
        """Load video frames with temporal augmentation."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None

            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                logger.error(f"No frames read from video: {video_path}")
                return None
                
            # Convert to numpy array and normalize
            frames = np.array(frames)
            frames = frames / 255.0  # Normalize to [0, 1]
            
            # Temporal augmentation: randomly select a sequence of frames
            if len(frames) > 30:  # If video is longer than 30 frames
                start_idx = np.random.randint(0, len(frames) - 30)
                frames = frames[start_idx:start_idx + 30]
            elif len(frames) < 30:  # If video is shorter than 30 frames
                # Repeat frames to reach 30
                frames = np.repeat(frames, 30 // len(frames) + 1, axis=0)[:30]
            
            # Convert to tensor and rearrange dimensions
            # frames shape: (30, 224, 224, 3) -> (3, 30, 224, 224)
            frames = torch.from_numpy(frames).float()
            frames = frames.permute(3, 0, 1, 2)  # Move channels to first dimension
            
            return frames
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        # Load video frames
        frames = self._load_video(video_path)
        if frames is None:
            # Return a zero tensor if video loading fails
            return torch.zeros((3, 30, 224, 224)), label
        
        return frames, self.label_to_idx[label]

class SignLanguageModel(nn.Module):
    """3D CNN + LSTM model for sign language recognition."""
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        # 3D CNN for video processing
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256 * 28 * 28,  # Adjust based on your input size
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, frames, height, width)
        batch_size, _, frames, _, _ = x.size()
        # 3D CNN
        x = self.conv3d(x)
        # Reshape for LSTM
        x = x.view(batch_size, frames, -1)
        # LSTM
        lstm_out, _ = self.lstm(x)
        # Use last LSTM output for classification
        x = lstm_out[:, -1, :]
        # Classification
        x = self.classifier(x)
        return x

def visualize_model_architecture(model, input_shape=(1, 3, 30, 224, 224)):
    """Create a detailed visualization of the model architecture."""
    # Create a sample input
    x = torch.randn(input_shape)
    
    # Create a figure for the architecture visualization
    plt.figure(figsize=(15, 10))
    
    # Plot CNN layers
    plt.subplot(2, 1, 1)
    plt.title('3D CNN Architecture')
    cnn_output = model.conv3d(x)
    cnn_output = cnn_output.detach().numpy()
    plt.imshow(cnn_output[0, 0, :, :, :].mean(axis=0), cmap='viridis')
    plt.colorbar()
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    # Plot LSTM layers
    plt.subplot(2, 1, 2)
    plt.title('LSTM Architecture')
    lstm_input = cnn_output.reshape(input_shape[0], -1, 256 * 28 * 28)
    lstm_out, _ = model.lstm(torch.from_numpy(lstm_input))
    lstm_out = lstm_out.detach().numpy()
    plt.imshow(lstm_out[0, :, :], cmap='viridis')
    plt.colorbar()
    plt.xlabel('Time Steps')
    plt.ylabel('Hidden Units')
    
    plt.tight_layout()
    plt.savefig('reports/model_architecture.png')
    plt.close()

def create_detailed_confusion_matrix(cm, class_names=None):
    """Create a detailed confusion matrix visualization with class names."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    plt.title('Detailed Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/detailed_confusion_matrix.png')
    plt.close()

def train_model(model, train_loader, val_loader, num_epochs, device):
    """Train the model and track loss/accuracy for visualization."""
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    best_val_loss = float('inf')
    train_losses, val_losses, val_accuracies = [], [], []
    
    # For confusion matrix
    all_preds = []
    all_targets = []
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            try:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            except Exception as e:
                logger.error(f"Training error at epoch {epoch}, batch {batch_idx}: {e}")
                continue
            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                try:
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    
                    # Store predictions and targets for confusion matrix
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                except Exception as e:
                    logger.error(f"Validation error at epoch {epoch}: {e}")
                    continue
        
        epoch_time = time.time() - epoch_start
        logger.info(f'Epoch {epoch} completed in {epoch_time:.2f} seconds')
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        train_losses.append(train_loss / len(train_loader))
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
            # Create visualizations for the best model
            visualize_model_architecture(model)
            cm = confusion_matrix(all_targets, all_preds)
            create_detailed_confusion_matrix(cm)
    
    total_time = time.time() - start_time
    logger.info(f'Total training time: {total_time:.2f} seconds')
    
    return train_losses, val_losses, val_accuracies, cm, total_time

def main():
    """Main function to run training and generate visualizations."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Log system info
    logger.info(f'PyTorch version: {torch.__version__}')
    logger.info(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'CUDA device: {torch.cuda.get_device_name(0)}')
    
    # Create dataset and dataloaders
    try:
        msasl_path = os.path.join(os.getcwd(), 'data', 'msasl')
        logger.info(f"Loading data from: {msasl_path}")
        dataset = SignLanguageDataset(msasl_path)
        
        # Get number of unique classes and create class mapping
        class_labels = sorted(set(label for _, label in dataset.samples))
        class_mapping = {label: f"Sign_{i+1}" for i, label in enumerate(class_labels)}
        logger.info(f"Class mapping: {class_mapping}")
        
        # Use smaller batch size for small dataset
        batch_size = 2  # Small batch size for 10 videos
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        logger.info(f"Found {len(dataset)} samples in msasl directory")
        logger.info(f"Number of classes: {len(class_labels)}")
        
        # Initialize model with correct number of classes
        model = SignLanguageModel(num_classes=len(class_labels))
        model = model.to(device)
        
        # Train model with adjusted parameters
        train_losses, val_losses, val_accuracies, cm, total_time = train_model(
            model, train_loader, val_loader, num_epochs=100, device=device  # More epochs for small dataset
        )
        
        # Save training metrics with class mapping
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'confusion_matrix': cm,
            'total_training_time': total_time,
            'device_used': str(device),
            'batch_size': batch_size,
            'learning_rate': 0.0001,  # Smaller learning rate
            'num_epochs': 100,
            'num_classes': len(class_labels),
            'class_mapping': class_mapping
        }
        torch.save(metrics, 'reports/training_metrics.pt')
        
        # Create confusion matrix with class names
        create_detailed_confusion_matrix(cm, class_names=[class_mapping[label] for label in class_labels])
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return

if __name__ == '__main__':
    main() 
