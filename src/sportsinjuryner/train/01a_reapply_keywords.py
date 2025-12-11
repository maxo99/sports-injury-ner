
from sportsinjuryner.config import load_jsonl, save_jsonl, settings
from sportsinjuryner.train.constants import (
    INJURY_KEYWORDS,
    INJURY_VERBS,
    ORG_BLACKLIST,
    REPORTER_BLACKLIST,
    STATUS_KEYWORDS,
    STATUS_PREFIXES,
    TEAM_WHITELIST,
)
from sportsinjuryner.train.ner_utils import tag_keywords

FILES_TO_UPDATE = [
    settings.OUTPUT_TRAIN,
    settings.OUTPUT_DEV,
    # settings.GOLD_STANDARD, # Uncomment if you want to update gold standard too
]


def reapply_keywords(data):
    updated_count = 0
    for item in data:
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]

        # 1. Reset INJURY and STATUS tags to O
        # Also reset TEAM tags if they match the blacklist
        for i, tag in enumerate(ner_tags):
            if "INJURY" in tag or "STATUS" in tag:
                ner_tags[i] = "O"
            elif "TEAM" in tag:
                # Check if the underlying token is in the blacklist
                # Note: This is a simple check. For multi-token entities, we might need more logic,
                # but since our blacklist items are mostly single tokens or handled by the tokenizer,
                # checking the token itself or the full entity reconstruction is safer.
                # For now, let's check if the token matches any part of a blacklisted item.
                token_word = tokens[i]
                if any(token_word in bl_item for bl_item in ORG_BLACKLIST):
                    ner_tags[i] = "O"
            elif "PLAYER" in tag:
                # Check if the underlying token is in the reporter blacklist
                token_word = tokens[i].lower()
                # Simple check for single tokens or parts of names
                if any(token_word in bl_item.split() for bl_item in REPORTER_BLACKLIST):
                    ner_tags[i] = "O"

        # 2. Re-apply Status Keywords & Prefixes
        ner_tags = tag_keywords(tokens, ner_tags, STATUS_KEYWORDS, tag_type="STATUS")
        ner_tags = tag_keywords(tokens, ner_tags, STATUS_PREFIXES, tag_type="STATUS")

        # 3. Re-apply Injury Keywords & Verbs
        ner_tags = tag_keywords(tokens, ner_tags, INJURY_KEYWORDS, tag_type="INJURY")
        ner_tags = tag_keywords(tokens, ner_tags, INJURY_VERBS, tag_type="INJURY")

        # 4. Re-apply Team Whitelist
        ner_tags = tag_keywords(tokens, ner_tags, TEAM_WHITELIST, tag_type="TEAM")

        item["ner_tags"] = ner_tags
        updated_count += 1

    return data


def main():
    print("Starting keyword reapplication...")

    for filename in FILES_TO_UPDATE:
        if not filename.exists():
            print(f"Skipping {filename} (not found)")
            continue

        print(f"\nProcessing {filename}...")
        data = load_jsonl(filename)

        if not data:
            print("  No data found.")
            continue

        updated_data = reapply_keywords(data)
        save_jsonl(updated_data, filename)
        print("  Done.")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
