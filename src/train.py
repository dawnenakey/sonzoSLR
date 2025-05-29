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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignLanguageDataset(Dataset):
    """Custom Dataset for loading sign language videos and annotations."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        try:
            self.samples = self._load_samples()
        except Exception as e:
            logger.error(f"Error loading samples: {e}")
            self.samples = []
            raise
        
    def _load_samples(self):
        # Load video paths and annotations
        samples = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    annotation_path = video_path.replace('.mp4', '.json')
                    if os.path.exists(annotation_path):
                        samples.append((video_path, annotation_path))
        if not samples:
            logger.warning(f"No samples found in {self.data_dir}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, annotation_path = self.samples[idx]
        # Load video frames
        frames = self._load_video(video_path)
        # Load annotations
        annotations = self._load_annotations(annotation_path)
        return frames, annotations

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

def train_model(model, train_loader, val_loader, num_epochs, device):
    """Train the model and track loss/accuracy for visualization."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    best_val_loss = float('inf')
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(num_epochs):
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
        train_losses.append(train_loss / len(train_loader))
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
                except Exception as e:
                    logger.error(f"Validation error at epoch {epoch}: {e}")
                    continue
        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        logger.info(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        # Learning rate scheduling
        scheduler.step(val_loss)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            # Upload to S3
            try:
                s3 = boto3.client('s3')
                s3.upload_file(
                    'best_model.pth',
                    'spokhand-ml-dev',
                    f'models/{datetime.now().strftime("%Y%m%d_%H%M%S")}_model.pth'
                )
            except Exception as e:
                logger.warning(f"Could not upload model to S3: {e}")
    return train_losses, val_losses, val_accuracies

def main():
    """Main function to run training and generate visualizations."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    # Create dataset and dataloaders
    try:
        dataset = SignLanguageDataset('data/raw')
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
    except Exception as e:
        logger.error(f"Error setting up data loaders: {e}")
        return
    # Initialize model
    model = SignLanguageModel(num_classes=1000)  # Adjust based on your number of signs
    model = model.to(device)
    # Train model
    try:
        train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs=50, device=device)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return
    # --- Loss/Accuracy Curve Plot ---
    try:
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Training/Validation Loss and Accuracy')
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/loss_accuracy_curves.png')
        plt.show()
        logger.info('Saved loss/accuracy curves to reports/loss_accuracy_curves.png')
    except Exception as e:
        logger.error(f"Error plotting loss/accuracy curves: {e}")
    # --- Confusion Matrix Plot ---
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = output.argmax(dim=1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        cm = confusion_matrix(y_true, y_pred)
        labels = [str(i) for i in range(cm.shape[0])]  # Replace with class names if available
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("reports/confusion_matrix.png")
        plt.show()
        logger.info('Saved confusion matrix to reports/confusion_matrix.png')
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")

if __name__ == '__main__':
    main() 