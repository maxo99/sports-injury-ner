import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import git
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    ROOT_DIR: str = str(git.Repo(".", search_parent_directories=True).working_tree_dir)
    SRC_DIR: Path = BASE_DIR / "src"
    DATA_DIR: Path = SRC_DIR / "data"
    CONFIG_DIR: Path = BASE_DIR / "config"

    # Data Files
    INPUT_CSV: Path = DATA_DIR / "injuries_espn.csv"
    INPUT_JSON: Path = DATA_DIR / "feed.json"
    OUTPUT_TRAIN: Path = DATA_DIR / "train.jsonl"
    OUTPUT_DEV: Path = DATA_DIR / "dev.jsonl"
    OUTPUT_TEST: Path = DATA_DIR / "test.jsonl"
    GOLD_STANDARD: Path = DATA_DIR / "gold_standard.jsonl"
    KEYWORDS_FILE: Path = CONFIG_DIR / "keywords.yaml"

    # Model Config
    # Model used to generate initial "silver" labels (must be a pre-trained NER model)
    DATA_GEN_MODEL: str = "dslim/bert-large-NER"
    # Base model to fine-tune (usually a domain-specific Masked LM)
    TRAIN_BASE_MODEL: str = "microsoft/SportsBERT"

    SPLIT_RATIO: float = 0.8

    # Logging
    LOG_LEVEL: str = "INFO"

    # Hugging Face
    HF_API_KEY: str | None = None
    HF_REPO_NAME: str = "maxo99/sports-injury-ner"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()


def load_keywords() -> dict[str, Any]:
    """Loads keywords from the YAML configuration file."""
    if not settings.KEYWORDS_FILE.exists():
        # Fallback or raise error? For now, let's log and return empty
        # But since we don't have the logger setup inside this function easily without circular deps if we use setup_logging
        # We'll just raise or return empty.
        # Given this is critical config, raising is better, but let's stick to the previous behavior of raising.
        raise FileNotFoundError(f"Config file not found at {settings.KEYWORDS_FILE}")

    with open(settings.KEYWORDS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(name: str | None = None) -> logging.Logger:
    """
    Configures and returns a logger with standard formatting.
    """
    logger = logging.getLogger(name if name else "sports_injury_ner")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)

    return logger


def get_utc_now():
    return datetime.now(UTC)



def load_jsonl(filename):
    data = []
    if Path(filename).exists():
        with open(filename, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def save_jsonl(data, filename):
    print(f"Saving {len(data)} examples to {filename}...")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

