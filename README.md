---
language:
- en
license: mit
base_model: microsoft/SportsBERT
tags:
- ner
- sports
- injury
- token-classification
model-index:
- name: sports-injury-ner
  results: []
---

# Sports Injury NER

This is a fine-tuned Named Entity Recognition (NER) model for extracting sports injury information from news text.

## Model Description

The goal of this project is to fine-tune [microsoft/SportsBERT](https://huggingface.co/microsoft/SportsBERT) to extract `PLAYER`, `INJURY`, `STATUS`, and `TEAM` from sports news using a Weak Supervision approach.

## Intended Use

The model is designed to extract the following entities from sports news:

- `PLAYER`: Name of the injured player.
- `INJURY`: Type of injury (e.g., "hamstring", "concussion").
- `STATUS`: Injury status (e.g., "questionable", "out", "placed on IR").
- `TEAM`: Team name (e.g., "Packers", "New York Giants").

## Usage

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_id = "maxo99/sports-injury-ner"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

text = "Packers QB Aaron Rodgers is questionable with a toe injury."
results = nlp(text)
print(results)
```

## Performance

| Metric | Score |
| :--- | :--- |
| **F1 Score** | **94.98%** |
| Precision | 95.41% |
| Recall | 94.55% |
| Accuracy | 97.06% |

*Evaluated: Dec 2025.*

## Training Data

The model was fine-tuned using a **Weak Supervision + Active Learning** approach:

1. **Silver Label Generation**: Initial labels generated using `bert-large-NER` (for Player/Team) and keyword matching (for Injury/Status).
2. **Active Learning**:
    - **Uncertainty Sampling**: "Least Confidence" scoring to identify confusing examples.
    - **Conflict Detection**: Flagging examples where keywords disagree with model predictions.
    - **Human-in-the-loop**: Manual validation of high-value candidates to create a "Gold Standard".
3. **Fine-tuning**: `microsoft/SportsBERT` fine-tuned on the combined Silver + Gold dataset.

## Limitations

- **Entity Resolution Priority**: The data generation script resolves overlapping entities using a strict priority: `Metadata > Keywords > BERT NER`.
- **Domain Specificity**: The model is trained on a specific dataset of NFL/ESPN reports and may not generalize well to other sports or writing styles.

## Development

For instructions on how to reproduce the training pipeline, run tests, or contribute, please see [docs/development.md](docs/development.md).

### Quick Start (DVC)

This project uses [DVC](https://dvc.org/) to manage the data pipeline.

```bash
# 1. Install dependencies
uv sync

# 2. Run the full pipeline (Data Gen -> Validation -> Training)
dvc repro train_model
```
