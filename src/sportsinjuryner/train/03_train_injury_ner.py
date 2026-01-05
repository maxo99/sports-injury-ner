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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from sportsinjuryner import __version__
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
    # Hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per device during training.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Linear warmup over warmup_ratio fraction of total steps.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model to use. If None, uses settings.TRAIN_BASE_MODEL.",
    )

    args_cli = parser.parse_args()

    # ============================================================================
    # 0. MLFLOW & HF SETUP
    # ============================================================================
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Log version information
    logger.info(f"Sports Injury NER Version: {__version__}")
    mlflow.set_tag("version", __version__)

    if settings.HF_API_KEY:
        logger.info("Logging in to Hugging Face Hub...")
        login(token=settings.HF_API_KEY)
    else:
        logger.warning(
            "HF_API_KEY not found. Model upload might fail if not logged in."
        )

    # ============================================================================
    # 2. LOAD & MERGE DATASET
    # ============================================================================
    # We want to train on Silver Data + Gold Data (Active Learning).
    # But we must ensure we don't have duplicates (Silver vs Gold for same text).

    silver_train_path = settings.OUTPUT_TRAIN
    gold_path = settings.GOLD_STANDARD

    # Load Silver Data
    if not silver_train_path.exists():
        raise FileNotFoundError(f"Training data not found at {silver_train_path}")

    logger.info(f"Loading Silver training data from {silver_train_path}...")
    silver_dataset = load_dataset(
        "json", data_files=str(silver_train_path), split="train"
    )

    # Load Gold Data (if exists)
    gold_examples = []
    if gold_path.exists():
        logger.info(f"Loading Gold data from {gold_path}...")
        gold_dataset = load_dataset("json", data_files=str(gold_path), split="train")
        gold_examples = list(gold_dataset)
        logger.info(f"Found {len(gold_examples)} gold examples.")

    # Merge Logic
    if gold_examples:
        # Create a set of Gold texts for fast lookup
        # We use " ".join(tokens) as the unique key
        gold_texts = {" ".join(ex["tokens"]) for ex in gold_examples}

        # Filter Silver: Keep only if text is NOT in Gold
        logger.info("Filtering Silver data to remove conflicts with Gold...")
        original_len = len(silver_dataset)
        filtered_silver = silver_dataset.filter(
            lambda x: " ".join(x["tokens"]) not in gold_texts
        )
        logger.info(
            f"Removed {original_len - len(filtered_silver)} Silver examples that are now in Gold."
        )

        # Combine
        from datasets import concatenate_datasets

        combined_train = concatenate_datasets([filtered_silver, gold_dataset])
        combined_train = combined_train.shuffle(seed=42)  # Shuffle to mix Gold/Silver
        logger.info(
            f"Final Training Set: {len(combined_train)} examples ({len(filtered_silver)} Silver + {len(gold_dataset)} Gold)"
        )

        train_dataset = combined_train

        # Since Gold is now in Train, we MUST NOT use it for Validation (leakage).
        # Fallback to Dev or Test for validation.
        validation_file = str(settings.OUTPUT_DEV)
        logger.info(f"Using {validation_file} for validation (since Gold is in Train).")

    else:
        train_dataset = silver_dataset
        validation_file = str(settings.OUTPUT_DEV)

    # Load Validation
    logger.info(f"Loading Validation data from {validation_file}...")
    eval_dataset = load_dataset("json", data_files=validation_file, split="train")

    dataset = {"train": train_dataset, "validation": eval_dataset}

    # ============================================================================
    # 3. TOKENIZATION & ALIGNMENT
    # ============================================================================
    base_model_name = args_cli.base_model or settings.TRAIN_BASE_MODEL
    logger.info(f"Loading tokenizer for {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Helper to map over the dictionary or dataset
    def tokenize_dataset(ds):
        return ds.map(
            lambda x: tokenize_and_align_labels(x, tokenizer),
            batched=True,
            remove_columns=ds.column_names,  # Remove original columns to save space/avoid issues
        )

    tokenized_train = tokenize_dataset(dataset["train"])
    tokenized_val = tokenize_dataset(dataset["validation"])

    tokenized_datasets = {"train": tokenized_train, "validation": tokenized_val}

    # ============================================================================
    # 4. TRAINING SETUP
    # ============================================================================
    logger.info("Initializing model...")
    model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir="sports-injury-ner-model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args_cli.learning_rate,
        per_device_train_batch_size=args_cli.per_device_train_batch_size,
        num_train_epochs=args_cli.num_train_epochs,
        weight_decay=args_cli.weight_decay,
        warmup_ratio=args_cli.warmup_ratio,
        push_to_hub=True,
        hub_model_id=settings.HF_REPO_NAME,
        hub_private_repo=False,
        logging_steps=10,
        report_to=["mlflow", "tensorboard"],  # Enable both MLflow and TensorBoard
        run_name=f"sports-injury-ner-v{__version__}",  # Name for the run in MLflow with version
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
    )  # type: ignore

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
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
