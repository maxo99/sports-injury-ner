from huggingface_hub import HfApi

from sportsinjuryner.config import settings


def main():
    print(f"Updating Model Card for {settings.HF_REPO_NAME}...")

    if not settings.HF_API_KEY:
        print("Error: HF_API_KEY not found in settings.")
        return

    readme_path = settings.BASE_DIR / "README.md"
    if not readme_path.exists():
        print(f"Error: README.md not found at {readme_path}")
        return

    print(f"Reading content from {readme_path}...")
    with open(readme_path, encoding="utf-8") as f:
        content = f.read()

    api = HfApi(token=settings.HF_API_KEY)

    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=settings.HF_REPO_NAME,
        repo_type="model",
    )
    print("Successfully updated README.md on Hugging Face Hub.")


if __name__ == "__main__":
    main()
