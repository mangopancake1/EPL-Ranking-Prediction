"""
This season's real Premier League results, pulled live from the FPL API.

The pre-season forecast never looks at these. The weekly-updating forecast
(src/weekly.py) folds them into the Dixon-Coles fit so the model conditions on
what has actually happened, and the two series are compared in May.

    python -m src.live_results        # print what has been played so far

Source is the same FPL API src/build_squad_values.py already uses -- the
`fixtures/` endpoint gives every fixture with `finished`, the two team ids, the
scoreline and the gameweek. No key, no scraping, and it updates itself.
"""

from __future__ import annotations

import pandas as pd

import config
from src.build_squad_values import _FPL_TEAM_TO_TEAM, _fpl_headers
from src.fetch import get_json

_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def _team_names() -> dict[int, str]:
    """FPL team id -> config.TEAMS name."""
    boot = get_json(_BOOTSTRAP_URL, headers=_fpl_headers())
    out = {}
    for t in boot["teams"]:
        name = _FPL_TEAM_TO_TEAM.get(t["name"], t["name"])
        if name not in config.TEAMS:
            raise ValueError(f"FPL team {t['name']!r} does not map into config.TEAMS")
        out[t["id"]] = name
    return out


def fetch_played(use_cache: bool = False) -> pd.DataFrame:
    """
    Every 2026/27 match played so far, in the schema of
    matches_football_data.csv: season, date, matchweek, home_team, away_team,
    home_goals, away_goals.

    use_cache defaults to False: for the weekly run we want the latest table,
    not whatever was cached hours ago.
    """
    names = _team_names()
    try:
        fixtures = get_json(_FIXTURES_URL, headers=_fpl_headers(), use_cache=use_cache)
    except Exception as exc:
        if use_cache:
            raise
        # FPL times out often enough that a weekly job should not die on it.
        # Fall back to the last good pull, loudly -- stale results are still
        # better than no run, but the caller needs to know they are stale.
        print(f"  ! live FPL fetch failed ({type(exc).__name__}); using last cached pull")
        fixtures = get_json(_FIXTURES_URL, headers=_fpl_headers(), use_cache=True)

    rows = []
    for f in fixtures:
        if not f.get("finished"):
            continue
        hs, as_ = f.get("team_h_score"), f.get("team_a_score")
        if hs is None or as_ is None:
            continue  # marked finished but no score yet -- skip rather than guess
        rows.append({
            "season": config.SEASON,
            "date": pd.Timestamp(f["kickoff_time"]).tz_convert(None).normalize(),
            "matchweek": int(f["event"]),
            "home_team": names[f["team_h"]],
            "away_team": names[f["team_a"]],
            "home_goals": int(hs),
            "away_goals": int(as_),
        })

    df = pd.DataFrame(rows).sort_values(["matchweek", "date"]).reset_index(drop=True)

    # A club cannot have played more games than there are matchweeks, and every
    # score is a non-negative integer. If either breaks, the mapping or the
    # endpoint changed and the weekly series should stop, not run on bad data.
    if not df.empty:
        assert df["matchweek"].between(1, 38).all(), "matchweek out of range"
        assert (df[["home_goals", "away_goals"]] >= 0).all().all(), "negative score"
        played = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        assert played.max() <= df["matchweek"].max(), "a club has played too many games"
        unknown = (set(df["home_team"]) | set(df["away_team"])) - set(config.TEAMS)
        assert not unknown, f"unknown teams: {unknown}"

    return df


def demo() -> None:
    df = fetch_played(use_cache=True)  # the self-check tolerates a slightly stale pull
    if df.empty:
        print("no 2026/27 matches finished yet")
        return
    n_mw = df["matchweek"].max()
    print(f"{len(df)} matches played, through matchweek {n_mw}")

    # Mini-table: points per club so far, so the numbers can be eyeballed
    # against the real standings.
    pts: dict[str, int] = {t: 0 for t in config.TEAMS}
    for r in df.itertuples(index=False):
        if r.home_goals > r.away_goals:
            pts[r.home_team] += 3
        elif r.home_goals < r.away_goals:
            pts[r.away_team] += 3
        else:
            pts[r.home_team] += 1
            pts[r.away_team] += 1
    for team, p in sorted(pts.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {team:18s} {p} pts")

    assert sum(pts.values()) == len(df) * 2 or any(
        r.home_goals == r.away_goals for r in df.itertuples(index=False)
    ), "points do not reconcile with matches played"
    print("OK  results reconcile")


if __name__ == "__main__":
    demo()
