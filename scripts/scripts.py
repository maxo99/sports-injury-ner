import subprocess
import sys


def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


def dvc_load():
    """Runs 'dvc repro --force load_data'"""
    print("Running: dvc repro --force load_data")
    run_command("dvc repro --force load_data")


def dvc_validate():
    """Runs the validation script directly."""
    print("Running: uv run python src/sportsinjuryner/train/02_validate_ner_data.py")
    run_command("uv run python src/sportsinjuryner/train/02_validate_ner_data.py")


def dvc_generate():
    """Runs 'dvc repro generate_data'"""
    print("Running: dvc repro generate_data")
    run_command("dvc repro generate_data")


def dvc_train():
    """Runs 'dvc repro train_model'"""
    print("Running: dvc repro train_model")
    run_command("dvc repro train_model")


def dvc_reapply():
    """Runs 'uv run python src/sportsinjuryner/train/01a_reapply_keywords.py'"""
    print("Running: uv run python src/sportsinjuryner/train/01a_reapply_keywords.py")
    run_command("uv run python src/sportsinjuryner/train/01a_reapply_keywords.py")


def dvc_repro():
    """Runs 'dvc repro'"""
    print("Running: dvc repro")
    run_command("dvc repro")


def mlflow_ui():
    """Runs 'mlflow ui' with the correct backend store URI."""
    print("Running: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    run_command("mlflow ui --backend-store-uri sqlite:///mlflow.db")
