"""Incremental fetcher for football-data.co.uk match files.

- Finished seasons: downloaded once, then skipped.
- Current season: re-checked every run, saved only if it has changed.
- Seasons are worked out from today's date, so new seasons are picked up automatically.
"""

import time
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
LEAGUES = ["E0", "E1"]  # E0 = Premier League, E1 = Championship
FIRST_SEASON_START = 2018  # 2018/19
RAW_DIR = Path("data/raw")


def season_code(start_year: int) -> str:
    """Turn a season's start year into the site's code, e.g. 2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    """From July onwards, count the new season as current."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def all_seasons() -> list[str]:
    return [season_code(y) for y in range(FIRST_SEASON_START, current_season_start() + 1)]


def fetch(league: str, season: str) -> bytes | None:
    url = BASE_URL.format(season=season, league=league)
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return None  # not published yet, e.g. just before a new season starts
    response.raise_for_status()
    return response.content


def run() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seasons = all_seasons()
    current = seasons[-1]
    for league in LEAGUES:
        for season in seasons:
            out_path = RAW_DIR / f"{league}_{season}.csv"
            if out_path.exists() and season != current:
                continue  # finished season we already have
            content = fetch(league, season)
            time.sleep(1)  # be polite to a free site
            if content is None:
                print(f"Not published yet: {out_path.name}")
                continue
            if out_path.exists() and out_path.read_bytes() == content:
                print(f"No change: {out_path.name}")
                continue
            status = "Updated" if out_path.exists() else "New"
            out_path.write_bytes(content)
            print(f"{status}: {out_path.name} ({len(content):,} bytes)")


if __name__ == "__main__":
    run()