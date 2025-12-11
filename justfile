


default:
    just --list


pipeline:
    uv run dvc repro

update-model-cards:
    uv run python src/scripts/update_model_cards.py

mlflow-ui:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

tensorboard:
    uv run tensorboard --db sqlite:///tensorboard.db

# 00 Load Data
load-data:
    uv run dvc repro --force load_data

# 01 Generate Data
generate-data:
    uv run dvc repro generate_data

# 01a Reapply Keywords
reapply-keywords:
    uv run python src/sportsinjuryner/train/01a_reapply_keywords.py

# 02 Validate Data
validate-data:
    uv run python src/sportsinjuryner/train/02_validate_ner_data.py

# 03 Train Model
train-model:
    uv run dvc repro train_model

# 04 Active Learning
active-learning:
    uv run dvc repro --force active_learning

# 05 Analyze Errors
analyze-errors:
    uv run dvc repro analyze_errors


