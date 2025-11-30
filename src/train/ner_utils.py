import re


def tokenize(text):
    """
    Simple tokenizer that splits on whitespace and punctuation.
    Returns a list of tokens.
    """
    # This regex splits on whitespace but keeps punctuation as separate tokens
    return re.findall(r"[\w']+|[.,!?;]", text)


def find_sublist(sublist, main_list):
    """
    Finds the start index of a sublist within a main list.
    Returns -1 if not found.
    """
    if not sublist:
        return -1
    n = len(sublist)
    for i in range(len(main_list) - n + 1):
        if main_list[i : i + n] == sublist:
            return i
    return -1


def tag_keywords(tokens, ner_tags, keywords, tag_type="INJURY"):
    """
    Tags tokens with B-{tag_type}/I-{tag_type} based on a list of keywords/phrases.
    Prioritizes longer phrases to avoid partial matches (e.g. tagging 'ankle' in 'high ankle sprain').
    """
    tokens_lower = [t.lower() for t in tokens]

    # Sort keywords by length (number of words) descending
    # This ensures "high ankle sprain" is matched before "ankle"
    sorted_keywords = sorted(keywords, key=lambda x: len(x.split()), reverse=True)

    for keyword in sorted_keywords:
        kw_tokens = tokenize(keyword)
        kw_tokens_lower = [t.lower() for t in kw_tokens]

        # We need to find ALL occurrences, not just the first one
        # So we loop through the text
        search_start_index = 0
        while True:
            # Find match starting from search_start_index
            # We have to slice the list, which changes indices, so we add search_start_index back
            sub_tokens = tokens_lower[search_start_index:]
            match_relative_idx = find_sublist(kw_tokens_lower, sub_tokens)

            if match_relative_idx == -1:
                break

            match_abs_idx = search_start_index + match_relative_idx

            # Check if any token in this range is already tagged
            is_overlap = False
            for i in range(len(kw_tokens)):
                if ner_tags[match_abs_idx + i] != "O":
                    is_overlap = True
                    break

            # Only tag if completely free (no overlap with Player/Status or other Injury)
            if not is_overlap:
                ner_tags[match_abs_idx] = f"B-{tag_type}"
                for i in range(1, len(kw_tokens)):
                    ner_tags[match_abs_idx + i] = f"I-{tag_type}"

            # Advance search
            search_start_index = match_abs_idx + 1

    return ner_tags
