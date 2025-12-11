from sportsinjuryner.config import load_keywords

_keywords = load_keywords()

INJURY_KEYWORDS = _keywords.get("injury_keywords", [])
INJURY_SUFFIXES = _keywords.get("injury_suffixes", [])

# Generate combinations of Injury Keywords + Suffixes
# e.g. "knee" + "injury" -> "knee injury"
# We add these to INJURY_KEYWORDS so they are caught as single entities
_combined_injuries = set(INJURY_KEYWORDS)
for kw in INJURY_KEYWORDS:
    for suffix in INJURY_SUFFIXES:
        _combined_injuries.add(f"{kw} {suffix}")

INJURY_KEYWORDS = list(_combined_injuries)

STATUS_KEYWORDS = _keywords.get("status_keywords", [])
STATUS_PREFIXES = _keywords.get("status_prefixes", [])
INJURY_VERBS = _keywords.get("injury_verbs", [])
ORG_BLACKLIST = _keywords.get("org_blacklist", [])
TEAM_WHITELIST = _keywords.get("team_whitelist", [])
REPORTER_BLACKLIST = _keywords.get("reporter_blacklist", [])
