import os
from typing import Any

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from config import settings, setup_logging

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
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)

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

    # ============================================================================
    # 2. LOAD DATASET
    # ============================================================================
    data_files = {
        "train": str(settings.OUTPUT_TRAIN),
        "validation": str(settings.OUTPUT_DEV),
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
        push_to_hub=False,
        logging_steps=10,
        report_to="mlflow",  # Enable MLflow tracking
        run_name="sports-injury-ner-v1",  # Name for the run in MLflow
    )

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
    # trainer.train()  # Uncomment to actually run training
    logger.info("Training setup complete. Ready to run (uncomment trainer.train()).")


if __name__ == "__main__":
    main()
