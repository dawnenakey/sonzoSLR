import torch
from src.models.sign_language_model import SignLanguageModel

def test_sign_language_model_forward():
    model = SignLanguageModel(input_size=4, num_classes=10)
    dummy_input = torch.randn(2, 5, 4)  # batch_size=2, sequence_length=5, input_size=4
    output = model(dummy_input)
    assert output.shape == (2, 10) 