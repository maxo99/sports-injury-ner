import argparse
import os
from typing import Any

import evaluate
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from datasets import load_dataset
from huggingface_hub import login
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from sportsinjuryner.config import settings, setup_logging

logger = setup_logging(__name__)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
# Define your custom label list.
# "BIO" (Beginning, Inside, Outside) format is standard for NER.
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

# Create mappings for the model
id2label = dict(enumerate(LABEL_LIST))
label2id = {label: i for i, label in enumerate(LABEL_LIST)}


def compute_metrics(p) -> dict[str, float]:
    """
    Computes metrics for the trainer.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [
            LABEL_LIST[p]
            for (p, label_id) in zip(prediction, label, strict=False)
            if label_id != -100
        ]
        for prediction, label in zip(predictions, labels, strict=False)
    ]
    true_labels = [
        [
            LABEL_LIST[label_id]
            for (p, label_id) in zip(prediction, label, strict=False)
            if label_id != -100
        ]
        for prediction, label in zip(predictions, labels, strict=False)
    ]

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    if results is None:
        return {}

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def tokenize_and_align_labels(
    examples: dict[str, Any], tokenizer: Any
) -> dict[str, Any]:
    """
    This function handles the mismatch between words and sub-tokens.
    Example: "Mayfield" might become ["May", "##field"].
    We must assign the label "I-PLAYER" to both (or ignore the second).
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens (like [CLS], [SEP]) get -100 (ignored by loss function)
            if word_idx is None:
                label_ids.append(-100)
            # If it's the start of a new word, use the label map
            elif word_idx != previous_word_idx:
                # Handle potential unknown labels gracefully
                lab = label[word_idx]
                label_ids.append(label2id.get(lab, 0))  # Default to O if unknown
            # If it's a sub-token of the same word, we usually ignore it (-100)
            # or repeat the label. Ignoring is standard BERT practice.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def main():
    logger.info("Starting training pipeline...")

    parser = argparse.ArgumentParser(description="Train Sports Injury NER Model")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run data loading and model init but skip actual training.",
    )
    args_cli = parser.parse_args()

    # ============================================================================
    # 0. MLFLOW & HF SETUP
    # ============================================================================
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    if settings.HF_API_KEY:
        logger.info("Logging in to Hugging Face Hub...")
        login(token=settings.HF_API_KEY)
    else:
        logger.warning(
            "HF_API_KEY not found. Model upload might fail if not logged in."
        )

    # ============================================================================
    # 2. LOAD DATASET
    # ============================================================================
    # Determine validation file
    if settings.GOLD_STANDARD.exists():
        logger.info(f"Using gold standard for validation: {settings.GOLD_STANDARD}")
        validation_file = str(settings.GOLD_STANDARD)
    else:
        logger.warning(
            f"Gold standard not found at {settings.GOLD_STANDARD}. Falling back to {settings.OUTPUT_DEV}"
        )
        validation_file = str(settings.OUTPUT_DEV)

    data_files = {
        "train": str(settings.OUTPUT_TRAIN),
        "validation": validation_file,
    }

    # Check if files exist
    if not os.path.exists(data_files["train"]):
        raise FileNotFoundError(
            f"Training data not found at {data_files['train']}. Run convert_csv_to_ner_data.py first."
        )

    # Load JSONL data
    dataset = load_dataset("json", data_files=data_files)
    logger.info(f"Loaded dataset: {dataset}")

    # ============================================================================
    # 3. TOKENIZATION & ALIGNMENT
    # ============================================================================
    logger.info(f"Loading tokenizer for {settings.TRAIN_BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(settings.TRAIN_BASE_MODEL)

    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    # ============================================================================
    # 4. TRAINING SETUP
    # ============================================================================
    logger.info("Initializing model...")
    model = AutoModelForTokenClassification.from_pretrained(
        settings.TRAIN_BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir="sports-injury-ner-model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
        hub_model_id=settings.HF_REPO_NAME,
        hub_private_repo=False,
        logging_steps=10,
        report_to="mlflow",  # Enable MLflow tracking
        run_name="sports-injury-ner-v1",  # Name for the run in MLflow
    ) # type: ignore

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # ============================================================================
    # 5. RUN
    # ============================================================================
    logger.info("Starting training...")
    if args_cli.dry_run:
        logger.info("Dry run enabled. Skipping actual training loop.")
        return

    trainer.train()
    logger.info("Training complete.")

    # ============================================================================
    # 6. EVALUATION & ARTIFACTS
    # ============================================================================
    logger.info("Evaluating on validation set for artifacts...")
    predictions, labels, metrics = trainer.predict(tokenized_datasets["validation"])
    predictions = np.argmax(predictions, axis=2)

    # Flatten predictions and labels, ignoring -100
    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels, strict=False):
        for p, label_id in zip(prediction, label, strict=False):
            if label_id != -100:
                true_predictions.append(LABEL_LIST[p])
                true_labels.append(LABEL_LIST[label_id])

    # Generate Confusion Matrix
    cm = confusion_matrix(true_labels, true_predictions, labels=LABEL_LIST)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_LIST,
        yticklabels=LABEL_LIST,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save to reports directory (Git tracked)
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    logger.info(f"Saved confusion matrix to {cm_path}")

    # Save metrics to JSON
    import json

    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Log to MLflow
    # Trainer usually closes the run at the end of train().
    # We try to get the active run or the last one.
    run = mlflow.active_run()
    if not run:
        run = mlflow.last_active_run()

    if run:
        # We need to resume the run to log artifacts if it was closed
        with mlflow.start_run(run_id=run.info.run_id):
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(metrics_path)
            logger.info(f"Logged artifacts to MLflow run {run.info.run_id}")
    else:
        logger.warning(
            "No active or recent MLflow run found. Skipping artifact logging."
        )


if __name__ == "__main__":
    main()
