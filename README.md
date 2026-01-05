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

![GitHub tag version](https://img.shields.io/github/v/tag/maxo99/sports-injury-ner?label=version&logo=github&color=blue)
<a href="https://huggingface.co/maxo99/sports-injury-ner">
  <img
    alt="Hugging Face Downloads"
    src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fmaxo99%2Fsports-injury-ner%3Fexpand%255B%255D%3Ddownloads%26expand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=maxo99%2Fsports-injury-ner&color=blue&logo=huggingface">
</a>

<!-- [![uv](https://img.shields.io/badge/uv-python%20package%20manager-111827?logo=uv&logoColor=white)](https://docs.astral.sh/uv/) -->
<!-- [![Transformers](https://img.shields.io/badge/Transformers-huggingface-FF6F61?logo=transformers&logoColor=white)](https://huggingface.co/docs/transformers/index) -->
[![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DVC](https://img.shields.io/badge/DVC-data%20version%20control-4B4B4B?logo=dvc&logoColor=white)](https://dvc.org/)
[![MLFlow](https://img.shields.io/badge/MLFlow-experiment%20tracking-13B9FD?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-metrics%20%26%20evaluation-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/stable/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-training%20visualization-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)
[![seqeval](https://img.shields.io/badge/seqeval-NER%20metrics-1E88E5)](https://github.com/chakki-works/seqeval)

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

text = "Packers QB Aaron Rodgers is questionable with a knee injury."
results = nlp(text)
print(results)
```

## Performance

| Metric | Score |
| :--- | :--- |
| **F1 Score** | **96.31%** |
| Precision | 95.69% |
| Recall | 96.94% |
| Accuracy | 98.28% |

*Evaluated: 2025-12-21.*

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
