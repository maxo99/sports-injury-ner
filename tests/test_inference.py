from unittest.mock import MagicMock, patch

import pytest
import torch

from sportsinjuryner.inference.predict import InjuryNER


@pytest.fixture
def mock_model_and_tokenizer():
    with (
        patch(
            "sportsinjuryner.inference.predict.AutoTokenizer.from_pretrained"
        ) as mock_tok,
        patch(
            "sportsinjuryner.inference.predict.AutoModelForTokenClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        tokenizer = MagicMock()
        # Mock tokenizer output
        tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 1000, 1001, 102]]),
            "offset_mapping": torch.tensor([[[0, 0], [0, 5], [6, 11], [0, 0]]]),
        }
        tokenizer.convert_ids_to_tokens.return_value = [
            "[CLS]",
            "Player",
            "Name",
            "[SEP]",
        ]
        mock_tok.return_value = tokenizer

        model = MagicMock()
        # Mock model output logits (batch=1, seq=4, labels=9)
        # 0: O, 1: B-PLAYER, 2: I-PLAYER
        logits = torch.zeros(1, 4, 9)
        logits[0, 1, 1] = 10.0  # B-PLAYER
        logits[0, 2, 2] = 10.0  # I-PLAYER

        output = MagicMock()
        output.logits = logits
        model.return_value = output
        mock_model_cls.return_value = model

        yield mock_tok, mock_model_cls


def test_predict_entities(mock_model_and_tokenizer):
    ner = InjuryNER(model_path="dummy")
    text = "Player Name"

    results = ner.predict(text)

    assert len(results) == 1
    entity = results[0]
    assert entity["entity"] == "PLAYER"
    assert entity["text"] == "Player Name"
    assert entity["score"] > 0.9


def test_predict_no_entities(mock_model_and_tokenizer):
    mock_tok, mock_model_cls = mock_model_and_tokenizer

    # Override logits to be all O
    model = mock_model_cls.return_value
    logits = torch.zeros(1, 4, 9)
    logits[0, :, 0] = 10.0  # All O
    model.return_value.logits = logits

    ner = InjuryNER(model_path="dummy")
    results = ner.predict("Nothing here")

    assert len(results) == 0
