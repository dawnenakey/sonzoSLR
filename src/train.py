import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path

from models.sign_language_model import SignLanguageModel
from data.asl_lex_loader import get_asl_lex_dataloader, ASLLexDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train for one epoch
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        signs = batch['sign'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(signs.unsqueeze(1))
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate the model
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        float: Average loss for validation
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Get data
            signs = batch['sign'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(signs.unsqueeze(1))
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train(
    data_path: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the sign language model
    
    Args:
        data_path: Path to ASL-LEX data
        num_epochs: Number of epochs to train for
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on
    """
    # Create data loaders
    train_loader = get_asl_lex_dataloader(
        data_path,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = get_asl_lex_dataloader(
        data_path,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Dynamically determine input_size and num_classes
    dataset = ASLLexDataset(data_path)
    input_size = dataset.X.shape[1]
    num_classes = len(dataset.label_encoder.classes_)
    
    # Initialize model
    model = SignLanguageModel(
        input_size=input_size,
        num_classes=num_classes
    ).to(device)
    
    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            logger.info("Saved best model")

if __name__ == "__main__":
    data_path = "src/data/asl_lex"
    train(data_path) 