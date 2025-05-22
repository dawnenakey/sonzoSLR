import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from depthai import Pipeline, Device
import boto3
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
        
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_loader.dataset)
        
        logger.info(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
            # Upload to S3
            s3 = boto3.client('s3')
            s3.upload_file(
                'best_model.pth',
                'spokhand-ml-dev',
                f'models/{datetime.now().strftime("%Y%m%d_%H%M%S")}_model.pth'
            )

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Initialize OAK camera
    pipeline = Pipeline()
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(640, 480)
    cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    
    # Create dataset and dataloaders
    dataset = SignLanguageDataset('data/raw')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = SignLanguageModel(num_classes=1000)  # Adjust based on your number of signs
    model = model.to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=50, device=device)

if __name__ == '__main__':
    main() 