import pytest
from transformers import AutoTokenizer

from sportsinjuryner.train.ner_utils import (
    align_tokens_and_labels,
    find_keyword_offsets,
)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_find_keyword_offsets_exact_match():
    text = "Player has a high ankle sprain."
    keywords = ["high ankle sprain"]
    results = find_keyword_offsets(text, keywords, "INJURY")

    assert len(results) == 1
    assert results[0]["text"] == "high ankle sprain"
    assert results[0]["start"] == 13
    assert results[0]["end"] == 30


def test_find_keyword_offsets_overlap_priority():
    text = "high ankle sprain"
    keywords = ["ankle", "high ankle sprain"]  # "high ankle sprain" should win
    results = find_keyword_offsets(text, keywords, "INJURY")

    assert len(results) == 1
    assert results[0]["text"] == "high ankle sprain"


def test_find_keyword_offsets_case_insensitive():
    text = "High Ankle Sprain"
    keywords = ["high ankle sprain"]
    results = find_keyword_offsets(text, keywords, "INJURY")

    assert len(results) == 1
    assert results[0]["text"] == "High Ankle Sprain"


def test_align_tokens_simple(tokenizer):
    text = "knee sprain"
    entities = [{"start": 0, "end": 4, "label": "INJURY", "text": "knee"}]
    tokens, tags = align_tokens_and_labels(text, tokenizer, entities)

    # bert-base-uncased: ['knee', 'sp', '##rain']
    assert tokens[0] == "knee"
    assert tags[0] == "B-INJURY"
    assert tags[1] == "O"


def test_align_tokens_subword_start(tokenizer):
    # "subword" -> "sub", "##word"
    # Entity: "word" (starts at index 3)
    text = "subword"
    entities = [{"start": 3, "end": 7, "label": "SUFFIX", "text": "word"}]
    tokens, tags = align_tokens_and_labels(text, tokenizer, entities)

    print(f"Tokens: {tokens}")
    print(f"Tags: {tags}")

    # Expect at least 2 tokens
    assert len(tokens) >= 2
    # sub (0-3) -> O
    assert tags[0] == "O"
    # ##word (3-7) -> B-SUFFIX
    assert tags[1] == "B-SUFFIX"
