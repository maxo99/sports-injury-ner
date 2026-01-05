import asyncio
import json
from datetime import datetime

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from sportsinjuryner.config import get_utc_now, settings, setup_logging

logger = setup_logging(__name__)


headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

RSS_FEEDS = {
    "NFL": {
        "DraftSharks": "https://www.draftsharks.com/rss/injury-news",
        "CBS Sports": "https://www.cbssports.com/rss/headlines/nfl/",
        "ESPN": "https://www.espn.com/espn/rss/nfl/news",
    },
    "NBA": {
        "RealGM": "https://basketball.realgm.com/rss/wiretap/0/0.xml",
        "CBS Sports": "https://www.cbssports.com/rss/headlines/nba/",
        "ESPN": "https://www.espn.com/espn/rss/nba/news",
    },
    "MLB": {
        "CBS Sports": "https://www.cbssports.com/rss/headlines/mlb/",
        "ESPN": "https://www.espn.com/espn/rss/mlb/news",
    },
    "NHL": {
        "Daily Faceoff": "https://www.dailyfaceoff.com/feed/",
        "CBS Sports": "https://www.cbssports.com/rss/headlines/nhl/",
        "ESPN": "https://www.espn.com/espn/rss/nhl/news",
    },
}


class FeedData(BaseModel):
    feed_id: str
    id: str
    collected_at: datetime = Field(default_factory=get_utc_now)
    title: str
    summary: str
    published: str
    link: str
    author: str = "Unknown"
    teams: list[str] = Field(default_factory=list)
    players: list[str] = Field(default_factory=list)

    @classmethod
    def from_feedparserdict(cls, feed_id: str, data: dict) -> "FeedData":
        return cls(
            feed_id=feed_id,
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            published=str(data.get("published", "")),
            author=str(data.get("author", "Unknown")),
            link=str(data.get("link", "")),
            teams=[],
            players=[],
            collected_at=get_utc_now(),
        )


async def collect_feeddatas(sports: list[str] | str = "all") -> list[FeedData]:
    fd_list = []

    if sports == "all":
        target_sports = RSS_FEEDS.keys()
    elif isinstance(sports, str):
        target_sports = [sports]
    else:
        target_sports = sports

    for sport in target_sports:
        if sport not in RSS_FEEDS:
            logger.warning(f"Sport {sport} not found in RSS_FEEDS")
            continue

        feeds = RSS_FEEDS[sport]
        for source, url in feeds.items():
            feed_id = f"{sport}-{source}"
            logger.info(f"Fetching {feed_id} from {url}")
            feed = await asyncio.to_thread(feedparser.parse, url)
            if feed.bozo:
                logger.error(f"Error fetching {feed_id}: {feed.bozo_exception}")
                continue
            logger.info(f"Fetched {len(feed.entries)} entries from {feed_id}")
            for entry in feed.entries:
                if isinstance(entry, dict):
                    try:
                        fd_list.append(FeedData.from_feedparserdict(feed_id, entry))
                    except Exception as e:
                        logger.error(f"Error processing {feed_id} entry {entry}: {e}")
                        continue
            logger.info(f"Collected {len(fd_list)} entries from {feed_id}")
    return fd_list


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data_to_save, f, indent=2, default=str)

        logger.info(f"Saved {len(data_to_save)} RSS entries to {output_path}")
    except Exception as e:
        logger.error(f"Failed to fetch RSS data: {e}")

