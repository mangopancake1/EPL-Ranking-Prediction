"""
Pull Premier League match results from football-data.org (free tier).

Goals only -- this tier has no xG. dixon_coles.fit_blended() already falls
back to a goals-only fit when xG columns are absent, so nothing downstream
needs to special-case it.

Requires FOOTBALL_DATA_TOKEN in the environment. Free tier: 10 req/min,
trailing ~3 seasons only (older seasons 403 regardless of competition).
"""

from __future__ import annotations

import os
import time

import pandas as pd

import config
from src.fetch import get_json

BASE = config.FOOTBALL_DATA_BASE


def _headers() -> dict:
    token = os.environ.get(config.FOOTBALL_DATA_TOKEN_ENV)
    if not token:
        raise RuntimeError(
            f"Set {config.FOOTBALL_DATA_TOKEN_ENV} in the environment "
            "(register free at football-data.org/client/register)"
        )
    return {"X-Auth-Token": token}


def fetch_season(season_label: str) -> pd.DataFrame:
    """One season of Premier League matches as a flat DataFrame."""
    year = config.FOOTBALL_DATA_SEASONS[season_label]
    url = f"{BASE}/competitions/PL/matches?season={year}"
    data = get_json(url, headers=_headers())

    rows = []
    for m in data.get("matches", []):
        if m["status"] != "FINISHED":
            continue
        score = m["score"]["fullTime"]
        rows.append(
            {
                "season": season_label,
                "date": m["utcDate"][:10],
                "matchweek": m["matchday"],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": score["home"],
                "away_goals": score["away"],
            }
        )
    return pd.DataFrame(rows)


def normalise_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """football-data.org uses full names ('Manchester United FC'); map to canonical."""
    lookup = {
        alias: canonical
        for canonical, aliases in {**config.TEAM_ALIASES, **config.NON_2026_27_TEAM_ALIASES}.items()
        for alias in aliases
    }
    # football-data.org appends "FC"/"AFC" inconsistently -- strip it before matching.
    def resolve(name: str) -> str:
        if name in lookup:
            return lookup[name]
        stripped = name
        for junk in (" FC", " AFC", "AFC "):
            stripped = stripped.replace(junk, "")
        stripped = stripped.strip()
        return lookup.get(stripped, name)

    df = df.copy()
    df["home_team"] = df["home_team"].map(resolve)
    df["away_team"] = df["away_team"].map(resolve)
    return df


def fetch_all(seasons: list[str] | None = None) -> pd.DataFrame:
    seasons = config.TRAIN_SEASONS if seasons is None else seasons
    frames = []
    for i, season in enumerate(seasons):
        print(f"  fetching {season} ...")
        frames.append(fetch_season(season))
        if i < len(seasons) - 1:
            time.sleep(config.REQUEST_DELAY["football-data"])
    df = pd.concat(frames, ignore_index=True)
    return normalise_team_names(df)


def fetch_fixtures_2026_27() -> pd.DataFrame:
    """The season being predicted -- scheduled, not yet played."""
    url = f"{BASE}/competitions/PL/matches?season={config.FOOTBALL_DATA_SEASONS['2026-2027']}"
    data = get_json(url, headers=_headers())
    rows = [
        {
            "date": m["utcDate"][:10],
            "matchweek": m["matchday"],
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "status": m["status"],
        }
        for m in data.get("matches", [])
    ]
    df = pd.DataFrame(rows)
    return normalise_team_names(df)


if __name__ == "__main__":
    matches = fetch_all()
    out = config.RAW / "matches_football_data.csv"
    matches.to_csv(out, index=False)

    assert len(matches) == 380 * len(config.TRAIN_SEASONS), (
        f"expected {380 * len(config.TRAIN_SEASONS)} matches, got {len(matches)}"
    )
    known = set(config.TEAMS) | set(config.NON_2026_27_TEAM_ALIASES)
    seen = set(matches["home_team"]) | set(matches["away_team"])
    unmapped = seen - known
    assert not unmapped, f"unmapped/unresolved team names: {unmapped}"
    assert matches[["home_goals", "away_goals"]].notna().all().all(), "missing scores"

    # Every 2026/27 club must actually appear somewhere in the training window,
    # or the model has no data to fit its attack/defence parameters from.
    current_20 = set(config.TEAMS)
    no_history = current_20 - seen
    if no_history:
        print(f"  ! no EPL match history in {config.TRAIN_SEASONS} for: {sorted(no_history)}")
        print("    (expected for newly promoted clubs -- they get squad-value-based strength instead)")

    fixtures = fetch_fixtures_2026_27()
    fixtures.to_csv(config.RAW / "fixtures_2026_27.csv", index=False)
    assert len(fixtures) == 380, f"expected 380 fixtures for 2026/27, got {len(fixtures)}"
    unmapped_fx = (set(fixtures["home_team"]) | set(fixtures["away_team"])) - current_20
    assert not unmapped_fx, f"unmapped fixture team names: {unmapped_fx}"

    print(f"OK  {len(matches)} matches across {config.TRAIN_SEASONS} -> {out}")
    print(f"OK  {len(fixtures)} fixtures for 2026-2027 -> fixtures_2026_27.csv")
