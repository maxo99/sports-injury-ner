import argparse
import json
from pathlib import Path
from typing import Any

from sportsinjuryner.config import settings, setup_logging
from sportsinjuryner.train.ner_utils import apply_reporter_filter

logger = setup_logging(__name__)

TAG_MAP = {
    "0": "O",
    "1": "B-PLAYER",
    "2": "I-PLAYER",
    "3": "B-INJURY",
    "4": "I-INJURY",
    "5": "B-STATUS",
    "6": "I-STATUS",
    "7": "B-TEAM",
    "8": "I-TEAM",
}


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


def print_example(tokens: list[str], tags: list[str]):
    print("\n" + "=" * 50)
    print(f"{'TOKEN':<20} {'TAG':<15}")
    print("-" * 35)
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        print(f"{i:<3} {token:<20} {tag:<15}")
    print("=" * 50)


def validate_example(example: dict[str, Any]) -> dict[str, Any] | str | None:
    tokens = example["tokens"]
    tags = example["ner_tags"]

    while True:
        print_example(tokens, tags)
        print("\nOptions:")
        print("  [enter] Accept as is")
        print("  [e]     Edit a tag")
        print("  [s]     Skip")
        print("  [q]     Quit and save")

        choice = input("Choice: ").strip().lower()

        if choice == "":
            return example
        elif choice == "s":
            return None
        elif choice == "q":
            return "QUIT"
        elif choice == "e":
            try:
                idx_input = input(
                    f"Enter token index (0-{len(tokens) - 1}) or range (e.g. 0-3): "
                )

                # Check for range
                if "-" in idx_input:
                    start_str, end_str = idx_input.split("-")
                    start_idx = int(start_str)
                    end_idx = int(end_str)

                    if 0 <= start_idx <= end_idx < len(tokens):
                        print("\nSelect Entity Type for Range:")
                        print("  0: O (Clear)")
                        print("  1: PLAYER")
                        print("  2: INJURY")
                        print("  3: STATUS")
                        print("  4: TEAM")

                        type_choice = input("Choice (0-4): ").strip()

                        if type_choice == "0":
                            for i in range(start_idx, end_idx + 1):
                                tags[i] = "O"
                        elif type_choice in ["1", "2", "3", "4"]:
                            entity_map = {
                                "1": "PLAYER",
                                "2": "INJURY",
                                "3": "STATUS",
                                "4": "TEAM",
                            }
                            entity_type = entity_map[type_choice]

                            # Set B- tag for first token
                            tags[start_idx] = f"B-{entity_type}"
                            # Set I- tag for the rest
                            for i in range(start_idx + 1, end_idx + 1):
                                tags[i] = f"I-{entity_type}"
                        else:
                            print("Invalid entity choice.")
                    else:
                        print("Invalid range indices.")
                # Handle single index (existing logic)
                elif idx_input.isdigit():
                    idx = int(idx_input)
                    if 0 <= idx < len(tokens):
                        print("\nAvailable tags:")
                        for key, val in TAG_MAP.items():
                            print(f"  {key}: {val}")

                        tag_input = (
                            input(
                                f"Enter new tag for '{tokens[idx]}' (0-8 or full tag): "
                            )
                            .strip()
                            .upper()
                        )

                        if tag_input in TAG_MAP:
                            tags[idx] = TAG_MAP[tag_input]
                        elif tag_input in TAG_MAP.values():
                            tags[idx] = tag_input
                        else:
                            print("Invalid tag.")
                    else:
                        print("Invalid index.")
                else:
                    print("Invalid input format.")
            except ValueError:
                print("Invalid input.")


def main():
    parser = argparse.ArgumentParser(description="Validate NER Data")
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(settings.OUTPUT_DEV),
        help="Path to the input file to validate (default: dev.jsonl)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    logger.info(f"Loading {input_path}...")
    data = load_data(input_path)

    if not data:
        logger.error(
            f"No data found in {input_path}. Run convert_csv_to_ner_data.py or active_learning.py first."
        )
        return

    gold_data = []
    if settings.GOLD_STANDARD.exists():
        logger.info(f"Loading existing gold data from {settings.GOLD_STANDARD}...")
        gold_data = load_data(settings.GOLD_STANDARD)
        logger.info(f"Loaded {len(gold_data)} existing gold examples.")

    # Filter out examples already in gold set (simple check by text content)
    existing_texts = {" ".join(item["tokens"]) for item in gold_data}
    to_review = [d for d in data if " ".join(d["tokens"]) not in existing_texts]

    print(f"Starting validation session. {len(to_review)} examples to review.")

    for i, example in enumerate(to_review):
        print(f"\nReviewing example {i + 1}/{len(to_review)}")

        # Show active learning score if available
        if "_al_score" in example:
            print(
                f"Active Learning Score: {example['_al_score']:.2f} (Conf: {example.get('_al_metrics', {}).get('confidence', 0):.2f})"
            )

        # Apply heuristic filter
        example["ner_tags"] = apply_reporter_filter(
            example["tokens"], example["ner_tags"]
        )

        result = validate_example(example)

        if result == "QUIT":
            break
        elif result:
            # Clean up temporary AL fields before saving to gold
            if "_al_score" in result:
                del result["_al_score"]
            if "_al_metrics" in result:
                del result["_al_metrics"]
            if "active_learning_score" in result:
                del result["active_learning_score"]
            if "meta" in result and "confidence" in result["meta"]:
                del result["meta"]  # Clean up meta if it was just for AL

            gold_data.append(result)
            # Auto-save after each valid entry
            save_data(gold_data, settings.GOLD_STANDARD)

    logger.info(f"Session ended. Total gold examples: {len(gold_data)}")
    logger.info(f"Saved to {settings.GOLD_STANDARD}")


if __name__ == "__main__":
    main()
