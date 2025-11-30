# Project Goal: Custom Sports Injury NER

## Objective

Fine-tune a BERT model to extract `PLAYER`, `INJURY`, `STATUS`, and `TEAM` from sports news.

## Workflow

1. **Generate Data** (`train/convert_csv_to_ner_data.py`)
    * Combines CSV (`injuries_espn.csv`) and JSON (`feed.json`) data.
    * Uses `dslim/bert-large-NER` for initial Player/Team detection.
    * Uses keyword matching for Injury and Status.
    * Outputs `data/train.jsonl` and `data/dev.jsonl`.

2. **Validate Data** (`train/validate_ner_data.py`)
    * Manual review of auto-generated tags to create a "Gold Standard".
    * Run: `uv run python train/validate_ner_data.py`

3. **Train Model** (`train/train_injury_ner.py`)
    * Fine-tunes `microsoft/SportsBERT` on the generated data.
    * Evaluates using `seqeval` (Precision, Recall, F1).
    * Run: `uv run python train/train_injury_ner.py`

## Current Status

* [x] Data generation script complete (CSV + JSON + Pre-trained NER + Keywords).
* [x] Validation script complete (with shortcuts).
* [x] Training script complete (loads real data).
* [ ] Run full training loop.

### 2. Extraction Strategies

* **LLM Analysis**: Explored `sport-injury-gemma2b-it-qlora` for QA-style extraction, but found the model artifact suspicious (46MB size).

## Key Files

* `main.py`: Main entry point for testing NER models.
* `train_injury_ner.py`: Reference implementation for fine-tuning SportsBERT (Gold Standard workflow).
* `hybrid_injury_extraction.py`: Prototype of the hybrid NER + Rules approach.
* `guide_finetune_injury_ner.py`: Documentation on the fine-tuning process.

## Next Steps for Future Agents

1. **Data Collection**: Gather a dataset of 500+ injury reports and annotate them with custom tags (`B-INJURY`, `I-STATUS`, etc.) to enable true fine-tuning.
2. **Pipeline Implementation**: Finalize the `hybrid_injury_extraction.py` script to process the live feed and output structured JSON.
3. **Evaluation**: Establish a "Golden Set" of manually verified injury reports to benchmark the accuracy of different extraction methods.
