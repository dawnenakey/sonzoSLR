import pytest
from src.data.asl_lex_loader import ASLLexDataset

def test_asl_lex_dataset_loading():
    dataset = ASLLexDataset('src/data/asl_lex')
    assert len(dataset) > 0
    sample = dataset[0]
    assert 'sign' in sample and 'label' in sample
    assert sample['sign'] is not None
    assert sample['label'] is not None 