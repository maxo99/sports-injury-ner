import asyncio
import logging
from datetime import UTC, datetime

import feedparser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _get_utc_now():
    return datetime.now(UTC)


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
    collected_at: datetime = Field(default_factory=_get_utc_now)
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
            collected_at=_get_utc_now(),
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
