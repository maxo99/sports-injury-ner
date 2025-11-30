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

# ============================================================================
# 1. CONFIGURATION (The "Scalable" Setup)
# ============================================================================
# We use SportsBERT because for a specialized task (injuries), starting with
# a model that knows the domain language is the robust, scalable choice.
MODEL_CHECKPOINT = "microsoft/SportsBERT"

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
]

# Create mappings for the model
id2label = dict(enumerate(LABEL_LIST))
label2id = {label: i for i, label in enumerate(LABEL_LIST)}

# ============================================================================
# 2. LOAD DATASET
# ============================================================================
data_files = {
    "train": "../data/train.jsonl",
    "validation": "../data/dev.jsonl",  # Or "../data/gold_standard.jsonl" if ready
}

# Check if files exist
import os

if not os.path.exists(data_files["train"]):
    raise FileNotFoundError(
        f"Training data not found at {data_files['train']}. Run convert_csv_to_ner_data.py first."
    )

# Load JSONL data
# The 'datasets' library handles JSONL natively
dataset = load_dataset("json", data_files=data_files)

# If no validation set found, split the training set
if "validation" not in dataset:
    print("No validation file found. Splitting training data...")
    dataset = dataset["train"].train_test_split(test_size=0.2)


# ============================================================================
# 3. TOKENIZATION & ALIGNMENT (The "Hard Part")
# ============================================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def tokenize_and_align_labels(examples):
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
                label_ids.append(label2id[label[word_idx]])
            # If it's a sub-token of the same word, we usually ignore it (-100)
            # or repeat the label. Ignoring is standard BERT practice.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)


# ============================================================================
# 4. METRICS (The "Right Evaluation")
# ============================================================================
# We use seqeval, the standard for NER. It calculates F1 score per entity type.
seqeval = evaluate.load("seqeval")


def compute_metrics(p):
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

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# ============================================================================
# 5. TRAINING SETUP
# ============================================================================
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(LABEL_LIST),
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    "sports-injury-ner-model",
    evaluation_strategy="epoch",  # Evaluate every epoch
    learning_rate=2e-5,  # Standard BERT learning rate
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_steps=1,  # Log often for this demo
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# ============================================================================
# 6. RUN
# ============================================================================
print("Starting training...")
# trainer.train()  # Uncomment to actually run training
print("Training setup complete. Ready to run.")
