import csv
import json
import zlib
from typing import Any

from transformers import AutoTokenizer, pipeline

from sportsinjuryner.config import settings, setup_logging
from train.constants import INJURY_KEYWORDS, ORG_BLACKLIST, STATUS_KEYWORDS
from train.ner_utils import align_tokens_and_labels, find_keyword_offsets

logger = setup_logging(__name__)

# Initialize NER pipeline (only once)
logger.info(f"Loading NER model ({settings.DATA_GEN_MODEL})...")
ner_pipeline = pipeline(
    "ner", model=settings.DATA_GEN_MODEL, aggregation_strategy="simple"
)  # type : ignore

# Initialize Tokenizer (for alignment)
logger.info(f"Loading Tokenizer ({settings.TRAIN_BASE_MODEL})...")
tokenizer = AutoTokenizer.from_pretrained(settings.TRAIN_BASE_MODEL)


def get_bert_ner_entities(text: str) -> list[dict[str, Any]]:
    """
    Runs the pre-trained NER model and returns entities with offsets.
    Maps PER -> PLAYER, ORG -> TEAM (if not blacklisted).
    """
    results = ner_pipeline(text)
    entities = []

    for entity in results:
        word = entity["word"].strip()
        entity_group = entity["entity_group"]
        start = entity["start"]
        end = entity["end"]

        label = None
        if entity_group == "PER":
            label = "PLAYER"
        elif entity_group == "ORG":
            if word in ORG_BLACKLIST:
                continue
            label = "TEAM"

        if label:
            entities.append(
                {"start": start, "end": end, "label": label, "text": text[start:end]}
            )

    return entities


def process_text(
    text: str, meta_entities: list[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Process a single text string:
    1. Find BERT NER entities
    2. Find Keyword entities (Injury, Status)
    3. Merge with provided meta_entities (Player, Team from metadata)
    4. Resolve overlaps
    5. Tokenize and align
    """
    if meta_entities is None:
        meta_entities = []

    # 1. BERT NER
    bert_entities = get_bert_ner_entities(text)

    # 2. Keywords
    injury_entities = find_keyword_offsets(text, INJURY_KEYWORDS, "INJURY")
    status_entities = find_keyword_offsets(text, STATUS_KEYWORDS, "STATUS")

    # 3. Combine all entities
    # Priority: Metadata > Keywords > BERT NER
    # We want to keep Metadata entities (Ground Truth) and Keywords (Specific Domain)
    # over generic BERT NER if they overlap.

    all_entities = meta_entities + injury_entities + status_entities + bert_entities

    # 4. Resolve Overlaps
    # We resolve overlaps by iterating through the prioritized list (Metadata > Keywords > BERT).
    # The first entity to claim a character span "wins".
    final_entities = []
    occupied = set()

    for ent in all_entities:
        start, end = ent["start"], ent["end"]
        is_overlap = False
        for i in range(start, end):
            if i in occupied:
                is_overlap = True
                break

        if not is_overlap:
            final_entities.append(ent)
            for i in range(start, end):
                occupied.add(i)

    # 5. Tokenize and Align
    tokens, ner_tags = align_tokens_and_labels(text, tokenizer, final_entities)

    return {"tokens": tokens, "ner_tags": ner_tags}


def process_csv() -> list[dict[str, Any]]:
    logger.info(f"Reading {settings.INPUT_CSV}...")
    data = []

    if not settings.INPUT_CSV.exists():
        logger.warning(f"{settings.INPUT_CSV} not found.")
        return data

    with open(settings.INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comment = row.get("Comment", "").strip()
            name = row.get("Name", "").strip()
            status = row.get("Status", "").strip()

            if not comment or not name:
                continue

            # Find metadata entities in text
            meta_entities = []

            # Player Name
            # We use find_keyword_offsets to find the name in the text
            name_ents = find_keyword_offsets(comment, [name], "PLAYER")
            if not name_ents and " " in name:
                # Try last name
                last_name = name.split()[-1]
                name_ents = find_keyword_offsets(comment, [last_name], "PLAYER")
            meta_entities.extend(name_ents)

            # Status from column
            if status:
                status_ents = find_keyword_offsets(comment, [status], "STATUS")
                meta_entities.extend(status_ents)

            result = process_text(comment, meta_entities)
            result["meta"] = {"player": name, "status": status, "source": "csv"}
            data.append(result)

    logger.info(f"Processed {len(data)} valid examples from CSV.")
    return data


def process_json() -> list[dict[str, Any]]:
    logger.info(f"Reading {settings.INPUT_JSON}...")
    data = []

    if not settings.INPUT_JSON.exists():
        logger.warning(f"{settings.INPUT_JSON} not found.")
        return data

    with open(settings.INPUT_JSON, encoding="utf-8") as f:
        try:
            feed_items = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding {settings.INPUT_JSON}")
            return data

    for item in feed_items:
        text = item.get("summary", "").strip()
        if not text:
            continue

        meta_entities = []

        # Players
        for player in item.get("players", []):
            if player:
                meta_entities.extend(find_keyword_offsets(text, [player], "PLAYER"))

        # Teams
        for team in item.get("teams", []):
            if team and team not in ORG_BLACKLIST:
                meta_entities.extend(find_keyword_offsets(text, [team], "TEAM"))

        result = process_text(text, meta_entities)
        result["meta"] = {"source": "json", "feed_id": item.get("feed_id")}
        data.append(result)

    logger.info(f"Processed {len(data)} valid examples from JSON.")
    return data


def save_jsonl(data: list[dict[str, Any]], filename: Any):
    logger.info(f"Saving {len(data)} examples to {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    # 1. Process the data
    csv_data = process_csv()
    json_data = process_json()

    all_data = csv_data + json_data

    # 2. Deterministic Split
    train_data = []
    dev_data = []
    test_data = []

    for item in all_data:
        # Use the text content for hashing to ensure stability
        # We reconstruct text from tokens for hashing purposes
        # (or we could use the original text if we had kept it, but tokens are fine)
        text_content = " ".join(item["tokens"])

        # Adler32 is fast and sufficient for this purpose
        h = zlib.adler32(text_content.encode("utf-8")) % 100

        if h < 80:
            train_data.append(item)
        elif h < 90:
            dev_data.append(item)
        else:
            test_data.append(item)

    logger.info(
        f"Split results: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)}"
    )

    # 4. Save
    save_jsonl(train_data, settings.OUTPUT_TRAIN)
    save_jsonl(dev_data, settings.OUTPUT_DEV)
    save_jsonl(test_data, settings.OUTPUT_TEST)

    # 5. Show a sample
    logger.info("Sample Output:")
    if train_data:
        sample = train_data[0]
        logger.info(f"Source: {sample['meta'].get('source')}")
        logger.info(f"Tokens: {sample['tokens']}")
        logger.info(f"Tags:   {sample['ner_tags']}")

        # Visual check
        logger.info("Visual Check:")
        for t, tag in zip(sample["tokens"], sample["ner_tags"]):
            logger.info(f"{t:15} {tag}")


if __name__ == "__main__":
    main()
