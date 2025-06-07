"""
Tests for the Sign Language Recognition model.
"""
import pytest
import torch
import numpy as np
from src.models.sign_language_model import SignLanguageModel

class TestSignLanguageModel:
    """Test suite for SignLanguageModel."""
    
    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        try:
            model = SignLanguageModel(num_classes=5)
            assert model is not None
            assert isinstance(model, SignLanguageModel)
        except Exception as e:
            pytest.fail(f"Model initialization failed: {str(e)}")

    def test_model_forward_pass(self, sample_input, sample_model):
        """Test model forward pass with sample input."""
        try:
            output = sample_model(sample_input)
            assert output is not None
            assert isinstance(output, torch.Tensor)
            assert output.shape[0] == sample_input.shape[0]  # batch size should match
            assert output.shape[1] == 5  # num_classes
        except Exception as e:
            pytest.fail(f"Forward pass failed: {str(e)}")

    def test_model_invalid_input(self, sample_model):
        """Test model behavior with invalid input."""
        with pytest.raises(Exception):
            # Test with wrong input shape
            invalid_input = torch.randn(1, 3, 20, 20)  # Wrong dimensions
            sample_model(invalid_input)

    @pytest.mark.integration
    def test_model_training_step(self, sample_input, sample_model):
        """Test a single training step."""
        try:
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(sample_model.parameters())
            
            # Create dummy target
            target = torch.randint(0, 5, (sample_input.shape[0],))
            
            # Forward pass
            output = sample_model(sample_input)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            assert loss.item() > 0
        except Exception as e:
            pytest.fail(f"Training step failed: {str(e)}")

    def test_model_device_transfer(self, sample_model, device):
        """Test model transfer to different devices."""
        try:
            model_on_device = sample_model.to(device)
            assert next(model_on_device.parameters()).device == device
        except Exception as e:
            pytest.fail(f"Device transfer failed: {str(e)}") 