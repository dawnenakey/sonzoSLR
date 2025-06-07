"""
Pytest configuration file for SpokHandSLR tests.
Contains common fixtures and configuration settings.
"""
import os
import pytest
import torch
from src.models.sign_language_model import SignLanguageModel

@pytest.fixture
def data_dirs():
    """Fixture to provide paths to existing data directories."""
    return {
        'msasl': os.path.join('data', 'msasl'),
        'wlasl': os.path.join('WLASL'),
        'asl_lex': os.path.join('src', 'data', 'asl_lex'),
        'raw': os.path.join('data', 'raw')
    }

@pytest.fixture
def sample_model():
    """Fixture to provide a sample model for testing."""
    model = SignLanguageModel(num_classes=5)
    return model

@pytest.fixture
def sample_input():
    """Fixture to provide sample input tensor for testing."""
    return torch.randn(1, 3, 30, 224, 224)  # batch_size=1, channels=3, frames=30, height=224, width=224

@pytest.fixture
def device():
    """Fixture to provide the device for testing (CPU/GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 