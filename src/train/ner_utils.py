import re
from typing import Any

from config import setup_logging

logger = setup_logging(__name__)


def find_keyword_offsets(
    text: str, keywords: list[str], label: str
) -> list[dict[str, Any]]:
    """
    Finds all non-overlapping occurrences of keywords in text and returns their character offsets.
    Prioritizes longer keywords.
    """
    entities = []
    text_lower = text.lower()

    # Sort keywords by length (descending) to prioritize longer phrases
    # e.g. match "high ankle sprain" before "ankle"
    sorted_keywords = sorted(keywords, key=len, reverse=True)

    # Keep track of occupied character indices to prevent overlaps
    occupied = set()

    for kw in sorted_keywords:
        if not kw:
            continue

        kw_lower = kw.lower()
        # Use word boundaries to avoid partial matches (e.g. "ache" in "headache")
        # Escape the keyword to handle special characters safely
        pattern = r"\b" + re.escape(kw_lower) + r"\b"

        try:
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()

                # Check if this range overlaps with any existing match
                is_overlap = False
                for i in range(start, end):
                    if i in occupied:
                        is_overlap = True
                        break

                if not is_overlap:
                    entities.append(
                        {
                            "start": start,
                            "end": end,
                            "label": label,
                            "text": text[start:end],
                        }
                    )
                    # Mark range as occupied
                    for i in range(start, end):
                        occupied.add(i)
        except re.error:
            logger.warning(f"Invalid regex pattern for keyword: {kw}")
            continue

    return entities


def align_tokens_and_labels(
    text: str, tokenizer: Any, entities: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """
    Tokenizes text using the provided tokenizer and aligns character-based entities to tokens.
    Returns (tokens, ner_tags).
    """
    # Tokenize with offsets
    tokenized = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
    offsets = tokenized["offset_mapping"]
    special_tokens_mask = tokenized["special_tokens_mask"]

    ner_tags = ["O"] * len(tokens)

    # Sort entities by start position
    entities = sorted(entities, key=lambda x: x["start"])

    for idx, (start, end) in enumerate(offsets):
        # Skip special tokens ([CLS], [SEP], etc.)
        if special_tokens_mask[idx]:
            continue

        # If start == end, it's usually a special token or empty, skip
        if start == end:
            continue

        # Check if this token falls within any entity
        for ent in entities:
            # We consider a token to be part of an entity if it starts within the entity boundaries
            # or if the entity covers the token.
            # Strict alignment: Token start must be >= Entity start AND Token end <= Entity end?
            # Or loose: Token overlaps?
            # BERT tokenization usually aligns well if we use the same text.

            # Logic:
            # If token start matches entity start -> B-TAG
            # If token is inside entity -> I-TAG

            if start >= ent["start"] and end <= ent["end"]:
                if start == ent["start"]:
                    ner_tags[idx] = f"B-{ent['label']}"
                else:
                    ner_tags[idx] = f"I-{ent['label']}"
                break
            elif start < ent["start"] and end > ent["start"]:
                # Token overlaps start of entity (rare with word boundaries, but possible)
                # e.g. "unbelievable" -> "un", "believ", "able"
                # If entity is "believable", "un" is outside.
                pass

    # Filter out special tokens
    final_tokens = []
    final_tags = []

    for idx, (start, end) in enumerate(offsets):
        if special_tokens_mask[idx]:
            continue

        final_tokens.append(tokens[idx])
        final_tags.append(ner_tags[idx])

    return final_tokens, final_tags
