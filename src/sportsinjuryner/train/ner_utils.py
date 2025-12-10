import re
from typing import Any

from sportsinjuryner.config import setup_logging

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
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
        is_split_into_words=False,  # We are providing a single string
    )

    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
    word_ids = tokenized.word_ids()

    # Create character-to-token mapping
    # We need to know which token corresponds to which character span
    # word_ids gives us the word index, but we need character offsets for the *original* string
    # Re-running with return_offsets_mapping to get character spans
    tokenized_with_offsets = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
    )
    offsets = tokenized_with_offsets["offset_mapping"]

    ner_tags = ["O"] * len(tokens)

    # Sort entities by start position
    entities = sorted(entities, key=lambda x: x["start"])

    for idx, (start, end) in enumerate(offsets):
        # Skip special tokens
        if word_ids[idx] is None:
            continue

        # Skip 0-length tokens
        if start == end:
            continue

        # Check overlap with entities
        for ent in entities:
            ent_start, ent_end = ent["start"], ent["end"]
            ent_label = ent["label"]

            # Strict alignment: Token must be substantially inside the entity
            # Or: if the token *starts* inside the entity, we tag it.

            # Case 1: Token starts exactly at entity start -> B-TAG
            if start == ent_start:
                ner_tags[idx] = f"B-{ent_label}"
                break

            # Case 2: Token is inside entity -> I-TAG
            elif start > ent_start and start < ent_end:
                ner_tags[idx] = f"I-{ent_label}"
                break

    # Filter out special tokens to match original behavior
    final_tokens = []
    final_tags = []

    for idx, token in enumerate(tokens):
        # word_ids[idx] is None for special tokens like [CLS], [SEP]
        if word_ids[idx] is None:
            continue

        final_tokens.append(token)
        final_tags.append(ner_tags[idx])

    return final_tokens, final_tags


def tag_keywords(
    tokens: list[str], tags: list[str], keywords: list[str], tag_type: str
) -> list[str]:
    """
    Applies keyword-based tagging to a list of tokens.
    Reconstructs text from tokens to handle multi-token keywords and subwords.
    """
    # 1. Build text and map characters to token indices
    text = ""
    char_to_token_idx = []

    for i, token in enumerate(tokens):
        if token.startswith("##"):
            clean_token = token[2:]
        else:
            if text:
                text += " "
                char_to_token_idx.append(None)  # Space
            clean_token = token

        text += clean_token
        char_to_token_idx.extend([i] * len(clean_token))

    # 2. Find keywords using the existing offset finder
    entities = find_keyword_offsets(text, keywords, tag_type)

    # 3. Apply tags
    for ent in entities:
        start, end = ent["start"], ent["end"]

        # Find all token indices covered by this entity
        covered_indices = set()
        for i in range(start, end):
            if i < len(char_to_token_idx):
                idx = char_to_token_idx[i]
                if idx is not None:
                    covered_indices.add(idx)

        sorted_indices = sorted(list(covered_indices))

        if not sorted_indices:
            continue

        # Apply B- tag to first token, I- to rest
        # Note: We overwrite existing tags. The caller should handle clearing if needed.
        tags[sorted_indices[0]] = f"B-{tag_type}"
        for idx in sorted_indices[1:]:
            tags[idx] = f"I-{tag_type}"

    return tags
