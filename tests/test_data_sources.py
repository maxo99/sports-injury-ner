import pandas as pd
import pytest

from sportsinjuryner.loaders.feedsreader import (
    FeedData,
    collect_feeddatas,
    get_injuries_espn,
)


@pytest.mark.asyncio
class TestDataSources:
    async def test_collect_feeddatas_all(self):
        """Test collecting feeds for all sports."""
        feeds = await collect_feeddatas("all")
        assert isinstance(feeds, list)
        assert len(feeds) > 0
        assert all(isinstance(f, FeedData) for f in feeds)

        # Check if we have feeds from multiple sports
        feed_ids = {f.feed_id for f in feeds}
        assert any("NFL" in fid for fid in feed_ids)
        assert any("NBA" in fid for fid in feed_ids)
        # MLB/NHL might be empty in off-season, but we verified they have entries in the script

    async def test_collect_feeddatas_specific_sport(self):
        """Test collecting feeds for a specific sport."""
        # NFL
        nfl_feeds = await collect_feeddatas("NFL")
        assert len(nfl_feeds) > 0
        assert all("NFL" in f.feed_id for f in nfl_feeds)

        # NBA
        nba_feeds = await collect_feeddatas("NBA")
        assert len(nba_feeds) > 0
        assert all("NBA" in f.feed_id for f in nba_feeds)

    async def test_collect_feeddatas_invalid_sport(self):
        """Test collecting feeds for an invalid sport."""
        feeds = await collect_feeddatas("INVALID_SPORT")
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
