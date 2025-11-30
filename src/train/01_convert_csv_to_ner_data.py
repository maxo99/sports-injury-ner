import csv
import json
import random

from pathlib import Path

from constants import INJURY_KEYWORDS, ORG_BLACKLIST, STATUS_KEYWORDS
from ner_utils import find_sublist, tag_keywords, tokenize
from transformers import pipeline

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_CSV = DATA_DIR / "injuries_espn.csv"
INPUT_JSON = DATA_DIR / "feed.json"
OUTPUT_TRAIN = DATA_DIR / "train.jsonl"
OUTPUT_DEV = DATA_DIR / "dev.jsonl"
SPLIT_RATIO = 0.8  # 80% train, 20% dev

# Initialize NER pipeline (only once)
print("Loading NER model (dslim/bert-large-NER)...")
ner_pipeline = pipeline(
    "ner", model="dslim/bert-large-NER", aggregation_strategy="simple"
)  # type : ignore


def apply_ner_model(text, tokens, ner_tags):
    """
    Runs the pre-trained NER model on the text and maps the results to our tokens.
    Updates ner_tags in place.
    """
    results = ner_pipeline(text)
    tokens_lower = [t.lower() for t in tokens]

    for entity in results:
        word = entity["word"].strip()
        entity_group = entity["entity_group"]

        # Map BERT labels to our labels
        if entity_group == "PER":
            tag_type = "PLAYER"
        elif entity_group == "ORG":
            # Check blacklist
            if word in ORG_BLACKLIST:
                continue
            tag_type = "TEAM"
        else:
            continue  # Skip LOC, MISC for now

        # Find this word in our tokens
        # Note: This is a fuzzy match because tokenizers differ.
        # We search for the entity text in our token list.
        entity_tokens = tokenize(word)
        entity_tokens_lower = [t.lower() for t in entity_tokens]

        start_idx = find_sublist(entity_tokens_lower, tokens_lower)

        if start_idx != -1:
            # Only tag if currently "O" (don't overwrite existing tags yet,
            # though usually this runs first so it sets the baseline)
            if ner_tags[start_idx] == "O":
                ner_tags[start_idx] = f"B-{tag_type}"
                for i in range(1, len(entity_tokens)):
                    if start_idx + i < len(ner_tags) and ner_tags[start_idx + i] == "O":
                        ner_tags[start_idx + i] = f"I-{tag_type}"

    return ner_tags


def process_csv():
    print(f"Reading {INPUT_CSV}...")

    data = []

    if not Path(INPUT_CSV).exists():
        print(f"Warning: {INPUT_CSV} not found.")
        return data

    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            comment = row.get("Comment", "").strip()
            name = row.get("Name", "").strip()
            status = row.get("Status", "").strip()

            # Skip rows without comments or names
            if not comment or not name:
                continue

            # 1. Tokenize the text
            tokens = tokenize(comment)
            tokens_lower = [t.lower() for t in tokens]

            # 2. Initialize tags as "O" (Outside)
            ner_tags = ["O"] * len(tokens)

            # 3. Apply Pre-trained NER Model (Baseline)
            ner_tags = apply_ner_model(comment, tokens, ner_tags)

            # 4. Tag the Player Name (Overwrite NER if needed, as this is ground truth)
            # Try full name first
            name_tokens = tokenize(name)
            name_tokens_lower = [t.lower() for t in name_tokens]
            start_idx = find_sublist(name_tokens_lower, tokens_lower)

            # If full name not found, try last name only (if name has multiple parts)
            if start_idx == -1 and len(name_tokens) > 1:
                last_name_tokens = [name_tokens_lower[-1]]
                start_idx = find_sublist(last_name_tokens, tokens_lower)

            if start_idx != -1:
                # Determine length of match (was it full name or last name?)
                match_len = (
                    len(name_tokens)
                    if find_sublist(name_tokens_lower, tokens_lower) != -1
                    else 1
                )

                ner_tags[start_idx] = "B-PLAYER"
                for i in range(1, match_len):
                    if start_idx + i < len(ner_tags):
                        ner_tags[start_idx + i] = "I-PLAYER"

            # 5. Tag the Status (if present in text)
            if status:
                status_tokens = tokenize(status)
                status_tokens_lower = [t.lower() for t in status_tokens]
                start_idx = find_sublist(status_tokens_lower, tokens_lower)

                if start_idx != -1:
                    ner_tags[start_idx] = "B-STATUS"
                    for i in range(1, len(status_tokens)):
                        if start_idx + i < len(ner_tags):
                            ner_tags[start_idx + i] = "I-STATUS"

            # 6. Tag Injury Keywords (Expanded Logic)
            ner_tags = tag_keywords(
                tokens, ner_tags, INJURY_KEYWORDS, tag_type="INJURY"
            )

            # 7. Add to dataset
            data.append(
                {
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    "meta": {"player": name, "status": status, "source": "csv"},
                }
            )

    print(f"Processed {len(data)} valid examples from CSV.")
    return data


def process_json():
    print(f"Reading {INPUT_JSON}...")

    data = []

    if not Path(INPUT_JSON).exists():
        print(f"Warning: {INPUT_JSON} not found.")
        return data

    with open(INPUT_JSON, encoding="utf-8") as f:
        try:
            feed_items = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding {INPUT_JSON}")
            return data

    for item in feed_items:
        # Use summary as the main text
        text = item.get("summary", "").strip()
        if not text:
            continue

        # 1. Tokenize
        tokens = tokenize(text)
        tokens_lower = [t.lower() for t in tokens]
        ner_tags = ["O"] * len(tokens)

        # 2. Apply Pre-trained NER Model (Baseline)
        ner_tags = apply_ner_model(text, tokens, ner_tags)

        # 3. Tag Players (if available in the 'players' list)
        players = item.get("players", [])
        for player_name in players:
            if not player_name:
                continue

            p_tokens = tokenize(player_name)
            p_tokens_lower = [t.lower() for t in p_tokens]
            start_idx = find_sublist(p_tokens_lower, tokens_lower)

            if start_idx != -1:
                ner_tags[start_idx] = "B-PLAYER"
                for i in range(1, len(p_tokens)):
                    if start_idx + i < len(ner_tags):
                        ner_tags[start_idx + i] = "I-PLAYER"

        # 4. Tag Teams (if available)
        teams = item.get("teams", [])
        for team in teams:
            if not team or team in ORG_BLACKLIST:
                continue

            # Teams might be abbreviations (CLE) or full names.
            # This simple check looks for exact matches of what's in the list.
            t_tokens = tokenize(team)
            t_tokens_lower = [t.lower() for t in t_tokens]
            start_idx = find_sublist(t_tokens_lower, tokens_lower)

            if start_idx != -1:
                ner_tags[start_idx] = "B-TEAM"
                for i in range(1, len(t_tokens)):
                    if start_idx + i < len(ner_tags):
                        ner_tags[start_idx + i] = "I-TEAM"

        # 5. Tag Status Keywords (New)
        ner_tags = tag_keywords(tokens, ner_tags, STATUS_KEYWORDS, tag_type="STATUS")

        # 6. Tag Injury Keywords (Expanded Logic)
        ner_tags = tag_keywords(tokens, ner_tags, INJURY_KEYWORDS, tag_type="INJURY")

        data.append(
            {
                "tokens": tokens,
                "ner_tags": ner_tags,
                "meta": {"source": "json", "feed_id": item.get("feed_id")},
            }
        )

    print(f"Processed {len(data)} valid examples from JSON.")
    return data


def save_jsonl(data, filename):
    print(f"Saving {len(data)} examples to {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    # 1. Process the data
    csv_data = process_csv()
    json_data = process_json()

    all_data = csv_data + json_data

    # 2. Shuffle
    random.seed(42)
    random.shuffle(all_data)

    # 3. Split
    split_idx = int(len(all_data) * SPLIT_RATIO)
    train_data = all_data[:split_idx]
    dev_data = all_data[split_idx:]

    # 4. Save
    save_jsonl(train_data, OUTPUT_TRAIN)
    save_jsonl(dev_data, OUTPUT_DEV)

    # 5. Show a sample
    print("\nSample Output:")
    if train_data:
        sample = train_data[0]
        print("Source:", sample["meta"].get("source"))
        print("Tokens:", sample["tokens"])
        print("Tags:  ", sample["ner_tags"])

        # Visual check
        print("\nVisual Check:")
        for t, tag in zip(sample["tokens"], sample["ner_tags"]):
            print(f"{t:15} {tag}")


if __name__ == "__main__":
    main()
