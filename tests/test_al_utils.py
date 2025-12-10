import pytest
import torch
from unittest.mock import MagicMock
from sportsinjuryner.train.al_utils import compute_metrics

@pytest.fixture
def mock_model():
    model = MagicMock()
    # Mock output logits: (batch_size=1, seq_len=3, num_labels=9)
    # Let's say we have 3 tokens.
    # Token 0: High confidence "O" (label 0)
    # Token 1: Low confidence "B-INJURY" (label 3)
    # Token 2: High confidence "O" (label 0)
    
    # Logits shape: (1, 3, 9)
    logits = torch.zeros(1, 3, 9)
    logits[0, 0, 0] = 10.0 # O
    logits[0, 1, 3] = 2.0  # B-INJURY (low confidence)
    logits[0, 1, 0] = 1.0  # O
    logits[0, 2, 0] = 10.0 # O
    
    output = MagicMock()
    output.logits = logits
    model.return_value = output
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 200, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "offset_mapping": torch.tensor([[[0, 0], [0, 5], [0, 0]]])
    }
    return tokenizer

def test_compute_metrics_low_confidence(mock_model, mock_tokenizer):
    device = torch.device("cpu")
    text = "knee injury"
    
    metrics = compute_metrics(mock_model, mock_tokenizer, text, device)
    
    # Check confidence
    # The lowest max prob should be for token 1.
    # Softmax([2.0, 1.0, ...]) -> exp(2)/(exp(2)+exp(1)+...)
    # It will be relatively low compared to the 10.0 ones.
    assert metrics["confidence"] < 0.9
    assert metrics["confidence"] > 0.0

def test_compute_metrics_conflict(mock_model, mock_tokenizer):
    # Mock model predicts NO injury (all O)
    # But text has injury keyword "sprain"
    
    device = torch.device("cpu")
    text = "ankle sprain"
    
    # Override model to predict all O
    logits = torch.zeros(1, 3, 9)
    logits[0, :, 0] = 10.0 # All O
    output = MagicMock()
    output.logits = logits
    mock_model.return_value = output
    
    metrics = compute_metrics(mock_model, mock_tokenizer, text, device)
    
    assert metrics["has_injury_kw"] is True # "sprain" is in keywords
    assert metrics["has_injury_pred"] is False
    assert metrics["conflict"] is True

def test_compute_metrics_no_conflict(mock_model, mock_tokenizer):
    # Mock model predicts INJURY
    # Text has injury keyword
    
    device = torch.device("cpu")
    text = "ankle sprain"
    
    # Override model to predict B-INJURY at index 1
    logits = torch.zeros(1, 3, 9)
    logits[0, 0, 0] = 10.0
    logits[0, 1, 3] = 10.0 # B-INJURY
    logits[0, 2, 0] = 10.0
    output = MagicMock()
    output.logits = logits
    mock_model.return_value = output
    
    metrics = compute_metrics(mock_model, mock_tokenizer, text, device)
    
    assert metrics["has_injury_kw"] is True
    assert metrics["has_injury_pred"] is True
    assert metrics["conflict"] is False
