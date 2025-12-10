import argparse

from sportsinjuryner.config import setup_logging
from sportsinjuryner.loaders.feedsreader import get_injuries_espn, get_rss_feed

logger = setup_logging(__name__)


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
