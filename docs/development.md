# Development Guide

## Workflow

This project uses **DVC** to manage the machine learning pipeline.

### 1. Data Pipeline

1. **Generate Data** (`src/sportsinjuryner/train/01_convert_csv_to_ner_data.py`)
    * Combines CSV (`src/data/injuries_espn.csv`) and JSON (`src/data/feed.json`) data.
    * Uses `dslim/bert-large-NER` for initial Player/Team detection.
    * Uses keyword matching for Injury and Status.
    * Outputs `src/data/train.jsonl`, `src/data/dev.jsonl`, and `src/data/test.jsonl`.
    * **Run**: `dvc repro generate_data`

2. **Active Learning Loop** (Optional but Recommended)
    * **Identify Candidates**: Run `src/sportsinjuryner/train/04_active_learning.py` to find uncertain or conflicting examples.
    * **Manual Validation**: Run `src/sportsinjuryner/train/02_validate_ner_data.py --input-file src/data/active_learning_candidates.jsonl`.
    * **Result**: Adds high-quality examples to `src/data/gold_standard.jsonl`.

3. **Train Model** (`src/sportsinjuryner/train/03_train_injury_ner.py`)
    * Fine-tunes `microsoft/SportsBERT`.
    * **Dynamic Merging**: Automatically merges `gold_standard.jsonl` into the training set at runtime.
    * **Validation**: Uses `dev.jsonl` (to prevent leakage since Gold is used for training).
    * **Run**: `dvc repro train_model`

4. **Evaluation** (`src/sportsinjuryner/train/05_error_analysis.py`)
    * Runs inference on the held-out `test.jsonl`.
    * Exports detailed error analysis to `reports/error_analysis.csv`.
    * **Run**: `uv run src/sportsinjuryner/train/05_error_analysis.py`

## Testing

* **Unit Tests**: Run the test suite to verify utilities and alignment logic.
  * Run: `uv run pytest`

## Experiment Tracking

* **MLflow UI**: View training metrics, parameters, and artifacts.
  * Run: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`
* **TensorBoard**: View real-time training dynamics (loss curves).
  * Run: `uv run tensorboard --logdir sports-injury-ner-model/runs`

## Key Files

* `src/sportsinjuryner/train/03_train_injury_ner.py`: Main training script (Hugging Face Trainer).
* `src/sportsinjuryner/train/04_active_learning.py`: Active Learning candidate generation.
* `src/sportsinjuryner/train/05_error_analysis.py`: Test set evaluation and error reporting.
* `dvc.yaml`: Pipeline definition.
