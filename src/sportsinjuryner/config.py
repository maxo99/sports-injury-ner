import logging
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    SRC_DIR: Path = BASE_DIR / "src"
    DATA_DIR: Path = SRC_DIR / "data"

    # Data Files
    INPUT_CSV: Path = DATA_DIR / "injuries_espn.csv"
    INPUT_JSON: Path = DATA_DIR / "feed.json"
    OUTPUT_TRAIN: Path = DATA_DIR / "train.jsonl"
    OUTPUT_DEV: Path = DATA_DIR / "dev.jsonl"
    OUTPUT_TEST: Path = DATA_DIR / "test.jsonl"
    GOLD_STANDARD: Path = DATA_DIR / "gold_standard.jsonl"

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
