# DVC Workflow Guide

This project uses [DVC (Data Version Control)](https://dvc.org/) to manage data pipelines and model versioning.

## Quick Start

### Running the Pipeline
To run the entire pipeline (data generation -> training):
```bash
uv run dvc repro
```

### Running Specific Stages
To run a specific stage (e.g., `generate_data`):
```bash
uv run dvc repro generate_data
```

### Forcing Re-execution
If DVC thinks a stage is up-to-date but you want to run it anyway (e.g., `load_data` or `validate_data` which are frozen):
```bash
uv run dvc repro --force <stage_name>
```

## Convenience Commands
We have added numbered shortcuts to `pyproject.toml` for the pipeline stages:

- **00-load**: `uv run 00-load` (Forces data loading)
- **01-generate**: `uv run 01-generate` (Runs data generation)
- **01a-reapply**: `uv run 01a-reapply` (Re-applies keywords to existing data)
- **02-validate**: `uv run 02-validate` (Runs manual validation)
- **03-train**: `uv run 03-train` (Runs training)
- **pipeline**: `uv run pipeline` (Runs full pipeline)

## Pipeline Stages

| Stage | Description | Input | Output |
| :--- | :--- | :--- | :--- |
| `load_data` | **(Frozen)** Runs `00_load_injuries_data.py` to fetch raw data. | Network/Scraper | `src/data/injuries_espn.csv`, `src/data/injuries_nfl.csv` |
| `generate_data` | Runs `01_convert_csv_to_ner_data.py` to create training data. | CSVs, `feed.json` | `src/data/train.jsonl`, `src/data/dev.jsonl` |
| `validate_data` | **(Frozen)** Runs `02_validate_ner_data.py` for manual review. | `dev.jsonl` | `src/data/gold_standard.jsonl` |
| `train_model` | Runs `03_train_injury_ner.py` to fine-tune the model. | `train.jsonl` | `sports-injury-ner-model/` |

## Managing Data
- **Tracking**: Large files and model artifacts are tracked by DVC, not Git.
- **Storage**: Local storage is configured at `/home/maxo/Work/dvc-storage`.
- **Versioning**: `dvc.lock` tracks the exact versions of data and models used. Commit this file to Git.
