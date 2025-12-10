import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from sportsinjuryner.config import settings, setup_logging
from sportsinjuryner.train.al_utils import compute_metrics
from sportsinjuryner.train.constants import REPORTER_BLACKLIST

logger = setup_logging(__name__)

# Define label list (must match training)
LABEL_LIST = [
    "O",
    "B-PLAYER",
    "I-PLAYER",
    "B-INJURY",
    "I-INJURY",
    "B-STATUS",
    "I-STATUS",
    "B-TEAM",
    "I-TEAM",
]

id2label = dict(enumerate(LABEL_LIST))
label2id = {label: i for i, label in enumerate(LABEL_LIST)}


def load_data(filename: Any) -> list[dict[str, Any]]:
    data = []
    if filename.exists():
        with open(filename, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def save_data(data: list[dict[str, Any]], filename: Any):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Active Learning: Sort Data by Uncertainty & Conflict"
    )
    parser.add_argument(
        "--model-path",
        default="sports-injury-ner-model",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(
            settings.OUTPUT_TRAIN
        ),  # Default to checking the training set for errors
        help="Path to the pool of data to analyze",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="src/data/active_learning_candidates.jsonl",
        help="Path to save the sorted candidates",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of output samples",
    )
    args = parser.parse_args()

    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {args.model_path} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_path,
            num_labels=len(LABEL_LIST),
            id2label=id2label,
            label2id=label2id,
        )
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 2. Load Data
    input_path = Path(args.input_file)
    logger.info(f"Loading data pool from {input_path}...")
    data_pool = load_data(input_path)

    if not data_pool:
        logger.error("No data found in pool.")
        return

    # 3. Filter & Score
    logger.info("Processing and scoring examples...")
    scored_items = []
    skipped_blacklist = 0

    for item in tqdm(data_pool):
        # Reconstruct text
        text = " ".join(item["tokens"]).replace(" ##", "")

        # Heuristic Filter: Skip blacklisted reporters
        if any(reporter.lower() in text.lower() for reporter in REPORTER_BLACKLIST):
            skipped_blacklist += 1
            continue

        metrics = compute_metrics(model, tokenizer, text, device)

        # Calculate Priority Score
        # Higher score = Higher priority for review
        # Priority = (Conflict * 1.0) + (1.0 - Confidence)
        # Conflict is boolean (0 or 1)
        # Confidence is 0.0 to 1.0
        # Max score ~ 2.0 (Conflict + Low Confidence)

        score = (1.0 if metrics["conflict"] else 0.0) + (1.0 - metrics["confidence"])

        item["_al_score"] = score
        item["_al_metrics"] = metrics
        scored_items.append(item)

    logger.info(f"Skipped {skipped_blacklist} items matching reporter blacklist.")

    # 4. Sort (Highest Score first)
    scored_items.sort(key=lambda x: x["_al_score"], reverse=True)

    # 5. Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.limit:
        scored_items = scored_items[: args.limit]

    logger.info(f"Saving top {len(scored_items)} candidates to {output_path}...")
    save_data(scored_items, output_path)

    # Print stats
    logger.info(f"Top 5 Candidates for Review:")
    for i in range(min(5, len(scored_items))):
        item = scored_items[i]
        text = " ".join(item["tokens"][:10]) + "..."
        metrics = item["_al_metrics"]
        logger.info(
            f"  {i + 1}. [Score: {item['_al_score']:.2f}] Conf: {metrics['confidence']:.2f}, Conflict: {metrics['conflict']} | {text}"
        )


if __name__ == "__main__":
    main()
