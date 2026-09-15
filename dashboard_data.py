"""
Shared data loader for every dashboard page.

Split out from the pages so caching happens once, not once per page. The
cache key is the newest mtime across the source files, so the moment
`run_pipeline` rewrites the outputs, the cache invalidates on its own --
there is no path to silently showing stale numbers.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts as _st_echarts

import config
from src import debutant_career, features


def st_echarts(options: dict, key: str | None = None, **kwargs):
    """st_echarts, but the key always reflects the option payload.

    The component only repaints when its Streamlit key changes identity --
    a fixed key (needed so a chart doesn't remount/flicker every rerun)
    means a chart driven by a selectbox/radio/filter silently keeps
    showing its first render forever, since the key never moves even
    though the data behind it did. Folding a hash of the options into the
    key gives it a fresh identity exactly when the data actually changes,
    and leaves it alone otherwise -- every interactive chart gets this for
    free just by importing st_echarts from here instead of the package.
    """
    digest = hashlib.md5(str(options).encode(), usedforsecurity=False).hexdigest()[:8]
    return _st_echarts(options, key=f"{key}_{digest}" if key else digest, **kwargs)

GOLD, TEAL, CORAL, PLOT, MUTED = "#C08C34", "#0F6F68", "#A83521", "#33506E", "#8595A9"
INK = "#131B29"
PALETTE = [GOLD, PLOT, TEAL, CORAL, "#7A5410", "#56C3B7"]

# Club colours, taken from each side's real 2026/27 kit and crest.
# Keys use config.TEAMS' canonical names, not the official long-form names.
CLUB_COLOURS = {
    # merah & marun
    "Arsenal":         ("#EF0107", "#FFFFFF"),
    "Aston Villa":     ("#7C2C3B", "#95BFE5"),
    "Bournemouth":     ("#E30613", "#000000"),
    "Brentford":       ("#E30613", "#FFFFFF"),
    "Liverpool":       ("#E51C25", "#F6EB61"),
    "Manchester Utd":  ("#C1040B", "#FFFFFF"),
    "Nott'ham Forest": ("#DD0000", "#FFFFFF"),
    "Sunderland":      ("#EB172B", "#FFFFFF"),
    # biru
    "Brighton":        ("#005DAA", "#FFFFFF"),
    "Chelsea":         ("#001489", "#FFFFFF"),
    "Coventry City":   ("#87CEEB", "#000000"),
    "Crystal Palace":  ("#0055A5", "#E30613"),
    "Everton":         ("#00009E", "#FFFFFF"),
    "Ipswich Town":    ("#0060A9", "#FFFFFF"),
    "Manchester City": ("#6CADDF", "#00285E"),
    # white, black, and other
    "Fulham":          ("#FFFFFF", "#000000"),
    "Hull City":       ("#FF8C00", "#000000"),
    "Leeds United":    ("#FFFFFF", "#FFE003"),
    "Newcastle Utd":   ("#231F20", "#FFFFFF"),
    "Tottenham":       ("#132257", "#FFFFFF"),
}

# White is invisible on a light background. For clubs whose primary colour is
# close to white, the secondary colour stands in as the chart marker instead
# -- Leeds reads as yellow, Fulham as black -- so every club still gets a
# colour that can actually be told apart, without inventing a new one.
_TOO_PALE = {"Fulham", "Leeds United"}


def club_colour(team: str) -> str:
    """One club's chart colour, already safe to use on a light background."""
    primary, secondary = CLUB_COLOURS.get(team, (PLOT, MUTED))
    return secondary if team in _TOO_PALE else primary


def club_colours(teams) -> list[str]:
    return [club_colour(t) for t in teams]


LINE_ID = {"GK": "Goalkeepers", "DEF": "Defence", "MID": "Midfield", "ATT": "Attack"}
LINES = list(LINE_ID.values())

_FILES = (
    config.OUTPUT / "prediction_2026_27.csv",
    config.OUTPUT / "rank_matrix_2026_27.csv",
    config.OUTPUT / "strengths_2026_27.csv",
    config.OUTPUT / "trajectory_2026_27.csv",
    config.OUTPUT / "simulations.npz",
    config.SQUAD_VALUES_FILE,
)


def stamp() -> float:
    return max(p.stat().st_mtime for p in _FILES if p.exists())


def require_outputs() -> None:
    missing = [p.name for p in _FILES if not p.exists()]
    if missing:
        st.error(
            f"Missing output files: {', '.join(missing)}. "
            "Run `python -m src.run_pipeline` first."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def _squads(_stamp: float) -> dict[str, pd.DataFrame]:
    from src.player_ratings import current_squad

    players = pd.DataFrame(
        json.loads(config.SQUAD_VALUES_FILE.read_text(encoding="utf-8"))["players"]
    )
    quality = debutant_career.quality_lookup()
    leagues = debutant_career.league_lookup()
    out = {}
    for team in config.TEAMS:
        # Same transfer-corrected roster the model scores, not the raw paste.
        sub = current_squad(players[players["club"] == team], team).copy()
        sub["club"] = team  # a priced arrival arrives with no club of its own
        sub["adaptation"] = [
            features.adaptation_factor(
                int(r.epl_matches or 0),
                leagues.get(getattr(r, "player", None), r.source_league),
                float(r.minutes_last_season or 0),
                player_quality=quality.get(getattr(r, "player", None)),
            )
            for r in sub.itertuples(index=False)
        ]
        sub["adj_value"] = sub["market_value_m"] * sub["adaptation"]
        out[team] = sub
    return out


@st.cache_data(show_spinner=False)
def _depth(_stamp: float) -> pd.DataFrame:
    squads = _squads(_stamp)
    tier1 = features.league_tier1_means(squads)
    rows = []
    for team in config.TEAMS:
        s = features.squad_strength(squads[team], tier1)
        row = {
            "team": team,
            "depth_ratio": s["depth_ratio"],
            "weakest_line": LINE_ID[s["weakest_line"]],
        }
        row.update({LINE_ID[k]: v for k, v in s["depth_by_line"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _merged(_stamp: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(config.OUTPUT / "prediction_2026_27.csv")
    grid = pd.read_csv(config.OUTPUT / "rank_matrix_2026_27.csv", index_col=0)
    audit = pd.read_csv(config.OUTPUT / "strengths_2026_27.csv")
    depth = _depth(_stamp)

    # depth_ratio exists on both sides; the recomputed one wins because it
    # carries the per-line breakdown too. Without this drop, pandas quietly
    # names them depth_ratio_x / _y and every reference on the pages below
    # ends up pointing at the wrong column.
    merged = (
        table.merge(audit.drop(columns=["depth_ratio"]), on=["team", "source"])
        .merge(depth, on="team")
    )
    assert "depth_ratio" in merged.columns
    assert len(merged) == len(config.TEAMS), "every club must appear exactly once"
    return table, grid, merged


@st.cache_data(show_spinner=False)
def _trajectory(_stamp: float) -> pd.DataFrame:
    """The table week by week, from monte_carlo.simulate()."""
    return pd.read_csv(config.OUTPUT / "trajectory_2026_27.csv")


@st.cache_data(show_spinner=False)
def _draws(_stamp: float) -> tuple[list[str], "np.ndarray"]:
    """
    All 10,000 raw simulation draws, not just their summary.

    This is what makes a boxplot or histogram possible: the CSV percentiles
    are only a handful of numbers taken from these arrays, and a distribution
    can't be redrawn from its own summary.
    """
    import numpy as np

    z = np.load(config.OUTPUT / "simulations.npz", allow_pickle=False)
    return list(z["teams"]), z["points"]


def load():
    """Return (table, grid, merged, squads) with caching already handled."""
    require_outputs()
    s = stamp()
    table, grid, merged = _merged(s)
    return table, grid, merged, _squads(s)


def trajectory() -> pd.DataFrame:
    require_outputs()
    return _trajectory(stamp())


def draws() -> tuple[list[str], "np.ndarray"]:
    require_outputs()
    return _draws(stamp())


@st.cache_data(show_spinner=False)
def _transfers() -> pd.DataFrame:
    """This summer's real transfers, one row per move, with the fee in €m."""
    from src.player_ratings import _transfers as parse

    rows = []
    for t in parse():
        rows.append({
            "from": t["from"], "to": t["to"], "player": t["player"],
            "fee_m": t["value_m"],
        })
    df = pd.DataFrame(rows)
    df["in_epl"] = df["to"].isin(config.TEAMS)
    df["out_epl"] = df["from"].isin(config.TEAMS)
    return df


@st.cache_data(show_spinner=False)
def _player_ratings(_stamp: float) -> pd.DataFrame:
    """Every squad player's real Premier League rating, where one exists."""
    from src import player_ratings as pr

    squads = _squads(_stamp)
    as_of = 2026
    rows = []
    for team, sub in squads.items():
        for r in sub.itertuples(index=False):
            rating = pr.player_rating(r.player, as_of)
            if rating is not None:
                rows.append({"team": team, "player": r.player, "position": r.position,
                             "market_value_m": r.market_value_m, "rating": rating})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _midweek_dates() -> dict[str, list[str]]:
    """Real EFL Cup / UCL / UEL / UECL dates per club (parse_european_schedule)."""
    path = config.DATA / "european_fixtures_2026_27.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _series_log(_mtime: float) -> pd.DataFrame:
    """Pre-season vs weekly-updating forecast, one row per club per run (src/weekly)."""
    path = config.OUTPUT / "series_log.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _prior_strength(_mtime: float) -> pd.DataFrame:
    """Matchdays to shift a club's relegation probability 10 points (src/prior_strength)."""
    path = config.OUTPUT / "prior_strength.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def series_log() -> pd.DataFrame:
    path = config.OUTPUT / "series_log.csv"
    return _series_log(path.stat().st_mtime if path.exists() else 0.0)


def prior_strength() -> pd.DataFrame:
    path = config.OUTPUT / "prior_strength.csv"
    return _prior_strength(path.stat().st_mtime if path.exists() else 0.0)


def transfers() -> pd.DataFrame:
    return _transfers()


def player_ratings_table() -> pd.DataFrame:
    require_outputs()
    return _player_ratings(stamp())


def midweek_dates() -> dict[str, list[str]]:
    return _midweek_dates()


def clubs_from_query(key: str, default: list[str]) -> list[str]:
    """Read the club selection from the URL, so this view can be shared as a link."""
    raw = st.query_params.get(key)
    if not raw:
        return default
    picked = [c for c in raw.split(",") if c in config.TEAMS]
    return picked or default


def sync_query(key: str, clubs: list[str]) -> None:
    if clubs:
        st.query_params[key] = ",".join(clubs)
    else:
        st.query_params.pop(key, None)
