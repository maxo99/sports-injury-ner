import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from sportsinjuryner.config import settings, setup_logging

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


def load_data(filename: Path) -> list[dict[str, Any]]:
    data = []
    if filename.exists():
        with open(filename, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def update_readme(metrics: dict[str, float]):
    """
    Updates the 'Performance' section of the README.md with the latest metrics.
    """
    readme_path = settings.BASE_DIR / "README.md"
    if not readme_path.exists():
        logger.warning(f"README.md not found at {readme_path}, skipping update.")
        return

    with open(readme_path, encoding="utf-8") as f:
        content = f.read()

    # Construct new performance section
    date_str = datetime.now().strftime("%Y-%m-%d")
    new_section = f"""## Performance

| Metric | Score |
| :--- | :--- |
| **F1 Score** | **{metrics['overall_f1']:.2%}** |
| Precision | {metrics['overall_precision']:.2%} |
| Recall | {metrics['overall_recall']:.2%} |
| Accuracy | {metrics['overall_accuracy']:.2%} |

*Evaluated: {date_str}.*
"""

    # Regex to replace existing Performance section
    # Matches ## Performance, then any content until the next ## header or end of string
    # We use a lookahead for the next header so we don't consume it
    pattern = r"(## Performance\n\n).*?(\n## |\Z)"
    
    if re.search(pattern, content, re.DOTALL):
        # Replace the content between the header and the next section
        new_content = re.sub(pattern, f"{new_section}\\2", content, flags=re.DOTALL, count=1)
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        logger.info("Updated README.md with new performance metrics.")
    else:
        logger.warning("Could not find '## Performance' section in README.md to update.")


def main():
    parser = argparse.ArgumentParser(description="Error Analysis on Test Set")
    parser.add_argument(
        "--model-path",
        default="sports-injury-ner-model",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(settings.OUTPUT_TEST),
        help="Path to the test data",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="reports/error_analysis.csv",
        help="Path to save the error report",
    )
    args = parser.parse_args()

    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {args.model_path} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        logger.info(f"Tokenizer type: {type(tokenizer)}")
        logger.info(f"Tokenizer is fast: {tokenizer.is_fast}")

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
    logger.info(f"Loading test data from {input_path}...")
    data = load_data(input_path)

    if not data:
        logger.error("No data found.")
        return

    # 3. Run Inference & Collect Metrics
    logger.info("Running inference...")

    all_true_labels = []
    all_pred_labels = []
    errors = []

    for example in tqdm(data):
        tokens = example["tokens"]
        true_tags = example["ner_tags"]

        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Get word_ids BEFORE converting to dict (which loses the method)
        word_ids = inputs.word_ids(0)

        # Move to device
        model_inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**model_inputs)

        logits = outputs.logits[0]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        # Align predictions
        if word_ids is None:
            continue

        aligned_preds = []
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                aligned_preds.append(preds[idx])
            previous_word_idx = word_idx

        aligned_preds = aligned_preds[: len(tokens)]

        if len(aligned_preds) != len(tokens):
            logger.warning(
                f"Length mismatch: Preds {len(aligned_preds)} vs Tokens {len(tokens)}"
            )
            logger.warning(f"Tokens: {tokens}")
            logger.warning(f"Word IDs: {word_ids}")
            continue

        # Convert IDs to Labels
        pred_labels = [id2label[p] for p in aligned_preds]

        # Store for metrics
        all_true_labels.append(true_tags)
        all_pred_labels.append(pred_labels)

        # Check for errors
        if pred_labels != true_tags:
            # Identify specific mismatches
            mismatches = []
            for i, (t, p) in enumerate(zip(true_tags, pred_labels)):
                if t != p:
                    mismatches.append(f"{tokens[i]} ({t} -> {p})")

            errors.append(
                {
                    "text": " ".join(tokens),
                    "mismatches": "; ".join(mismatches),
                    "true_tags": str(true_tags),
                    "pred_tags": str(pred_labels),
                }
            )

    # 4. Compute Metrics
    logger.info("Computing metrics...")
    logger.info(f"Collected {len(all_true_labels)} examples for evaluation.")

    if len(all_true_labels) == 0:
        logger.error("No examples collected! Check alignment logic.")
        return

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=all_pred_labels, references=all_true_labels)

    if results:
        logger.info("Test Set Results:")
        logger.info(f"  Precision: {results['overall_precision']:.4f}")
        logger.info(f"  Recall:    {results['overall_recall']:.4f}")
        logger.info(f"  F1 Score:  {results['overall_f1']:.4f}")
        logger.info(f"  Accuracy:  {results['overall_accuracy']:.4f}")

        # Save metrics to JSON
        metrics_path = Path(args.output_csv).parent / "test_metrics.json"
        with open(metrics_path, "w") as f:
            # Convert numpy types to float
            clean_results = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in results.items()
                if not isinstance(v, dict)
            }
            json.dump(clean_results, f, indent=2)

        # Update README
        update_readme(results)

    # 5. Save Errors
    if errors:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(errors)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(errors)} error examples to {output_csv}")
    else:
        logger.info("No errors found! (Perfect score?)")


if __name__ == "__main__":
    main()
