"""
Tests for data loading functionality in SpokHandSLR.
"""
import pytest
import os
from src.data.asl_lex_loader import ASLLexDataset

def test_asl_lex_dataset_loading(data_dirs):
    """Test ASL Lex dataset loading with actual data."""
    dataset = ASLLexDataset(data_dirs['asl_lex'])
    assert len(dataset) > 0
    sample = dataset[0]
    assert 'sign' in sample and 'label' in sample
    assert sample['sign'] is not None
    assert sample['label'] is not None

def test_data_directory_structure(data_dirs):
    """Test that all required data directories exist."""
    for dir_name, dir_path in data_dirs.items():
        assert os.path.exists(dir_path), f"Data directory {dir_name} not found at {dir_path}"

@pytest.mark.integration
def test_data_loading_integration(data_dirs):
    """Integration test for data loading from all sources."""
    # Test ASL Lex data
    asl_lex_dataset = ASLLexDataset(data_dirs['asl_lex'])
    assert len(asl_lex_dataset) > 0
    
    # Add more dataset loading tests as needed
    # For example, when you implement MSASL or WLASL loaders 