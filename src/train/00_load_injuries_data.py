import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import settings, setup_logging

logger = setup_logging(__name__)

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_injuries_espn():
    INJURIES_URL = "https://www.espn.com/nfl/injuries"
    logger.info(f"Fetching ESPN injuries from {INJURIES_URL}")

    try:
        response = requests.get(INJURIES_URL, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch ESPN data: {e}")
        return

    soup = BeautifulSoup(response.content, "html.parser")

    # Find all of the teams table scrollers div
    scrollers = soup.find_all("div", class_="Table__Scroller")

    if not scrollers:
        logger.warning(
            "No injury tables found on ESPN page. Layout might have changed."
        )
        return

    data = []
    for scroller in scrollers:
        table = scroller.find("table")
        if not table:
            continue

        tbody = table.find("tbody")
        if not tbody:
            continue

        rows = tbody.find_all("tr")

        for row in rows:
            cells = row.find_all("td")
            row_data = [cell.get_text(strip=True) for cell in cells]
            if row_data:
                data.append(row_data)

    # Create DataFrame and save to CSV
    columns = ["Name", "Position", "Date", "Status", "Comment"]
    df = pd.DataFrame(data, columns=columns)

    logger.info(f"Extracted {len(df)} rows from ESPN.")
    if not df.empty:
        logger.debug(f"Sample data:\n{df.head()}")

    output_path = settings.INPUT_CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")


def get_injuries_nfl():
    INJURIES_URL = "https://www.nfl.com/injuries/"
    logger.info(f"Fetching NFL injuries from {INJURIES_URL}")

    try:
        response = requests.get(INJURIES_URL, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch NFL data: {e}")
        return

    soup = BeautifulSoup(response.content, "html.parser")

    tables = soup.find_all("table")

    if not tables:
        logger.warning("No tables found on NFL.com page.")
        return

    data = []
    for table in tables:
        tbody = table.find("tbody")
        if not tbody:
            continue

        rows = tbody.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            row_data = [cell.get_text(strip=True) for cell in cells]
            if row_data:
                data.append(row_data)

    # Create DataFrame and save to CSV
    columns = ["Player", "Position", "Injuries", "Practice Status", "Game Status"]
    df = pd.DataFrame(data, columns=columns)

    logger.info(f"Extracted {len(df)} rows from NFL.com")
    if not df.empty:
        logger.debug(f"Sample data:\n{df.head()}")

    # Note: We might want to save this to a different file or merge,
    # but keeping original logic for now.
    output_path = settings.DATA_DIR / "injuries_nfl.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    logger.info("Starting injury data collection...")
    get_injuries_espn()
    get_injuries_nfl()
    logger.info("Data collection complete.")
