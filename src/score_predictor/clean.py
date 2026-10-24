"""Combine raw football-data files into one clean match table."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/matches.parquet")

LEAGUE_NAMES = {"E0": "EPL", "E1": "Championship"}

# Raw column name -> our column name.
# The odds columns (average closing odds) are kept for the market benchmark later.
COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "AvgCH": "odds_home",
    "AvgCD": "odds_draw",
    "AvgCA": "odds_away",
}


def load_file(path: Path) -> pd.DataFrame:
    league_code, season = path.stem.split("_")  # "E0_2526" -> "E0", "2526"
    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
    df = df.reindex(columns=list(COLUMNS)).rename(columns=COLUMNS)
    df["league"] = LEAGUE_NAMES[league_code]
    df["season"] = season
    return df


def build() -> pd.DataFrame:
    frames = [load_file(p) for p in sorted(RAW_DIR.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df[["home_goals", "away_goals"]] = df[["home_goals", "away_goals"]].astype(int)
    df = df.sort_values(["date", "league"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    matches = build()
    print(f"Saved {len(matches):,} matches to {OUT_PATH}")
    print(f"Date range: {matches['date'].min():%d %b %Y} to {matches['date'].max():%d %b %Y}\n")
    print(matches.groupby(["league", "season"]).size().unstack(level=0))