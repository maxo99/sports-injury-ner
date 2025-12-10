from pathlib import Path

import yaml

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "keywords.yaml"


def load_keywords():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_keywords = load_keywords()

INJURY_KEYWORDS = _keywords.get("injury_keywords", [])
STATUS_KEYWORDS = _keywords.get("status_keywords", [])
ORG_BLACKLIST = _keywords.get("org_blacklist", [])
TEAM_WHITELIST = _keywords.get("team_whitelist", [])
REPORTER_BLACKLIST = _keywords.get("reporter_blacklist", [])
