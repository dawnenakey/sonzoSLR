"""
ASL Sentence-Level Training Pipeline using ASLLRP Data

This module implements sequence-to-sequence training for ASL sentence recognition
using Boston University's ASLLRP dataset.

Author: SpokHand SLR Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import cv2
from datetime import datetime

# Import our ASLLRP integration
from asllrp_integration import ASLLRPDataLoader, ASLLRPDatasetConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLSentenceDataset(Dataset):
    """Dataset for ASL sentence-level training."""
    
    def __init__(self, training_data: List[Dict], video_root: str, max_sequence_length: int = 50):
        self.training_data = training_data
        self.video_root = Path(video_root)
        self.max_sequence_length = max_sequence_length
        
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        item = self.training_data[idx]
        
        # Load video frames
        video_frames = self._load_video_frames(item['video_path'])
        
        # Get input and target sequences
        input_sequence = torch.tensor(item['input_sequence'], dtype=torch.long)
        target_sequence = torch.tensor(item['target_sequence'], dtype=torch.long)
        
        # Pad sequences to max length
        input_sequence = self._pad_sequence(input_sequence)
        target_sequence = self._pad_sequence(target_sequence)
        
        return {
            'video_frames': video_frames,
            'input_sequence': input_sequence,
            'target_sequence': target_sequence,
            'video_id': item['video_id'],
            'sentence_id': item['sentence_id'],
            'timestamps': item['timestamps']
        }
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load video frames for the given sentence."""
        video_path = self.video_root / video_path
        
        if not video_path.exists():
            logger.warning(f"Video file not found: {video_path}")
            # Return dummy frames
            return torch.zeros(3, 30, 224, 224)
        
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0  # Normalize
            
            frames.append(frame)
        
        cap.release()
        
        # Convert to tensor and ensure we have enough frames
        frames = np.array(frames)
        if len(frames) < 30:
            # Repeat frames to reach minimum length
            frames = np.repeat(frames, 30 // len(frames) + 1, axis=0)[:30]
        elif len(frames) > 30:
            # Sample 30 frames evenly
            indices = np.linspace(0, len(frames) - 1, 30, dtype=int)
            frames = frames[indices]
        
        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)
        
        return frames
    
    def _pad_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Pad sequence to max length."""
        if len(sequence) >= self.max_sequence_length:
            return sequence[:self.max_sequence_length]
        
        # Pad with PAD token (assuming 0 is PAD token)
        padding = torch.zeros(self.max_sequence_length - len(sequence), dtype=torch.long)
        return torch.cat([sequence, padding])

class VideoEncoder(nn.Module):
    """Video encoder using 3D CNN + Transformer."""
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        
        # 3D CNN for video feature extraction
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
        
        # Transformer encoder for temporal modeling
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )
        
        # Projection layer
        self.projection = nn.Linear(256 * 28 * 28, d_model)
        
    def forward(self, video_frames):
        # video_frames: (batch, channels, frames, height, width)
        batch_size, _, frames, _, _ = video_frames.size()
        
        # 3D CNN
        conv_out = self.conv3d(video_frames)  # (batch, 256, frames, 28, 28)
        
        # Reshape for transformer
        conv_out = conv_out.view(batch_size, frames, -1)  # (batch, frames, 256*28*28)
        
        # Project to d_model
        conv_out = self.projection(conv_out)  # (batch, frames, d_model)
        
        # Transformer encoder
        # Add positional encoding
        pos_encoding = self._get_positional_encoding(frames, conv_out.size(-1))
        conv_out = conv_out + pos_encoding.unsqueeze(0)
        
        # Transformer encoding
        encoded = self.transformer_encoder(conv_out.transpose(0, 1))  # (frames, batch, d_model)
        
        return encoded.transpose(0, 1)  # (batch, frames, d_model)
    
    def _get_positional_encoding(self, seq_len: int, d_model: int):
        """Generate positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

class TextDecoder(nn.Module):
    """Transformer decoder for text generation."""
    
    def __init__(self, vocab_size: int, d_model: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, memory, tgt_mask=None):
        # tgt: (batch, seq_len)
        # memory: (batch, frames, d_model)
        
        # Embed target sequence
        tgt_embedded = self.embedding(tgt) * np.sqrt(self.d_model)
        
        # Add positional encoding
        seq_len = tgt.size(1)
        pos_encoding = self._get_positional_encoding(seq_len, self.d_model)
        tgt_embedded = tgt_embedded + pos_encoding.unsqueeze(0)
        
        # Transformer decoder
        decoded = self.transformer_decoder(
            tgt_embedded.transpose(0, 1),  # (seq_len, batch, d_model)
            memory.transpose(0, 1),        # (frames, batch, d_model)
            tgt_mask
        )
        
        # Output projection
        output = self.output_projection(decoded.transpose(0, 1))  # (batch, seq_len, vocab_size)
        
        return output
    
    def _get_positional_encoding(self, seq_len: int, d_model: int):
        """Generate positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

class ASLSentenceModel(nn.Module):
    """Complete ASL sentence recognition model."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, max_sequence_length: int = 50):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
        # Video encoder
        self.video_encoder = VideoEncoder(d_model)
        
        # Text decoder
        self.text_decoder = TextDecoder(vocab_size, d_model)
        
        # Special tokens
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.unk_token = 3
    
    def forward(self, video_frames, target_sequence=None):
        # Encode video
        video_features = self.video_encoder(video_frames)  # (batch, frames, d_model)
        
        if target_sequence is not None:
            # Training mode - use teacher forcing
            # Create target mask for decoder
            tgt_mask = self._generate_square_subsequent_mask(target_sequence.size(1))
            
            # Decode with teacher forcing
            output = self.text_decoder(target_sequence, video_features, tgt_mask)
            return output
        else:
            # Inference mode - generate sequence
            return self._generate_sequence(video_features)
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate mask for transformer decoder."""
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _generate_sequence(self, video_features, max_length=None):
        """Generate sequence during inference."""
        if max_length is None:
            max_length = self.max_sequence_length
        
        batch_size = video_features.size(0)
        device = video_features.device
        
        # Start with SOS token
        generated = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Create mask for current sequence
            tgt_mask = self._generate_square_subsequent_mask(generated.size(1))
            
            # Decode current sequence
            output = self.text_decoder(generated, video_features, tgt_mask)
            
            # Get next token
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have EOS token
            if (next_token == self.eos_token).all():
                break
        
        return generated

def train_sentence_model():
    """Train the ASL sentence recognition model."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load ASLLRP data
    logger.info("Loading ASLLRP dataset...")
    loader = ASLLRPDataLoader("/path/to/asllrp/data")  # Update path
    videos = loader.load_dataset()
    
    # Convert to training format
    converter = ASLLRPDatasetConverter(videos)
    training_data = converter.convert_to_training_format()
    
    # Save training data
    converter.save_training_data("asllrp_training_data.json")
    
    # Create dataset
    dataset = ASLSentenceDataset(training_data, "/path/to/asllrp/videos")
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    # Initialize model
    vocab_size = len(converter.vocabulary)
    model = ASLSentenceModel(vocab_size=vocab_size)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD tokens
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            video_frames = batch['video_frames'].to(device)
            input_sequence = batch['input_sequence'].to(device)
            target_sequence = batch['target_sequence'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(video_frames, input_sequence)
            
            # Compute loss
            loss = criterion(output.view(-1, vocab_size), target_sequence.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                video_frames = batch['video_frames'].to(device)
                input_sequence = batch['input_sequence'].to(device)
                target_sequence = batch['target_sequence'].to(device)
                
                output = model(video_frames, input_sequence)
                loss = criterion(output.view(-1, vocab_size), target_sequence.view(-1))
                val_loss += loss.item()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_asl_sentence_model.pth')
        
        logger.info(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, '
                   f'Val Loss: {val_loss/len(val_loader):.4f}')
    
    logger.info("Training completed!")

if __name__ == "__main__":
    train_sentence_model()
