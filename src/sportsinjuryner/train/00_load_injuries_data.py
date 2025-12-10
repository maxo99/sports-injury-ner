import argparse
import asyncio
import json

import pandas as pd
import requests
from bs4 import BeautifulSoup

from sportsinjuryner.config import settings, setup_logging
from sportsinjuryner.loaders.feedsreader import collect_feeddatas

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
    return df


def get_rss_feed():
    logger.info("Fetching RSS feeds...")
    try:
        feed_data = asyncio.run(collect_feeddatas())

        # Convert to list of dicts for JSON serialization
        data_to_save = [item.model_dump() for item in feed_data]

        output_path = settings.INPUT_JSON
        with open(output_path, "w") as f:
            json.dump(data_to_save, f, indent=2, default=str)

        logger.info(f"Saved {len(data_to_save)} RSS entries to {output_path}")
    except Exception as e:
        logger.error(f"Failed to fetch RSS data: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load injury data from various sources."
    )
    parser.add_argument(
        "--espn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch ESPN data",
    )
    parser.add_argument(
        "--nfl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fetch NFL data",
    )
    parser.add_argument(
        "--rss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch RSS feeds",
    )

    args = parser.parse_args()

    logger.info("Starting injury data collection...")

    if args.espn:
        get_injuries_espn()

    if args.rss:
        get_rss_feed()

    logger.info("Data collection complete.")
