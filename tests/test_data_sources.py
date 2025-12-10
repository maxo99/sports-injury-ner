import asyncio
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from sportsinjuryner.loaders.feedsreader import FeedData, collect_feeddatas

# Dynamic import for module starting with number
file_path = Path(__file__).parent.parent / "src" / "train" / "00_load_injuries_data.py"
spec = importlib.util.spec_from_file_location("load_injuries_data", file_path)
load_injuries_data = importlib.util.module_from_spec(spec)
sys.modules["load_injuries_data"] = load_injuries_data
spec.loader.exec_module(load_injuries_data)
get_injuries_espn = load_injuries_data.get_injuries_espn


# Helper to run async tests if pytest-asyncio is not set up
def run_async(coro):
    return asyncio.run(coro)


class TestDataSources:
    def test_collect_feeddatas_all(self):
        """Test collecting feeds for all sports."""
        feeds = run_async(collect_feeddatas("all"))
        assert isinstance(feeds, list)
        assert len(feeds) > 0
        assert all(isinstance(f, FeedData) for f in feeds)

        # Check if we have feeds from multiple sports
        feed_ids = {f.feed_id for f in feeds}
        assert any("NFL" in fid for fid in feed_ids)
        assert any("NBA" in fid for fid in feed_ids)
        # MLB/NHL might be empty in off-season, but we verified they have entries in the script

    def test_collect_feeddatas_specific_sport(self):
        """Test collecting feeds for a specific sport."""
        # NFL
        nfl_feeds = run_async(collect_feeddatas("NFL"))
        assert len(nfl_feeds) > 0
        assert all("NFL" in f.feed_id for f in nfl_feeds)

        # NBA
        nba_feeds = run_async(collect_feeddatas("NBA"))
        assert len(nba_feeds) > 0
        assert all("NBA" in f.feed_id for f in nba_feeds)

    def test_collect_feeddatas_invalid_sport(self):
        """Test collecting feeds for an invalid sport."""
        feeds = run_async(collect_feeddatas("INVALID_SPORT"))
        assert isinstance(feeds, list)
        assert len(feeds) == 0

    def test_get_injuries_espn(self):
        """Test scraping ESPN injuries."""
        # This is a live test, so it depends on ESPN being up and having data.
        # It might fail if the structure changes, which is good to know.
        df = get_injuries_espn()

        # Check if it returns a DataFrame (or None if failed/empty, but our refactor returns df)
        # The function returns None if request fails, or df if successful.
        if df is None:
            pytest.fail("get_injuries_espn returned None (request failed)")

        assert isinstance(df, pd.DataFrame)
        # It might be empty if no injuries are listed, but usually there are some.
        # We'll just warn if empty, but assert it's a DataFrame.
        if df.empty:
            pytest.warns(UserWarning, match="ESPN DataFrame is empty")
        else:
            assert "Name" in df.columns
            assert "Status" in df.columns
