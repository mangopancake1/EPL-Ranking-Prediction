"""
Championship-to-Premier-League offset for the promoted clubs' team layer.

    python -m src.championship        # self-check + what the fit currently says

The promoted clubs' level comes from promoted_baseline(): the average points per
game past promoted clubs actually managed. That is a population number -- every
promoted club starts at it, and only squad value and the rating blend separate
them. Their own Championship record is thrown away.

Managers and player ratings are already discounted by league (features.
league_weight); the team layer was the one place a lower-league record counted
for nothing. This module measures how much a club's Championship
points-per-game in its promotion season says about its Premier League
points-per-game the season after, using every past promoted club whose 'after'
is in matches_football_data.csv, and applies that line to the current three.

    pl_ppg = a + b * champ_ppg          fitted on n past promoted clubs

Two sources feed the fit. data/championship_records.json holds a handful of
hand-collected FootyStats seasons (n=8) -- at that n the sign of the slope is
noise, and the high end is parachute clubs bouncing straight back. The second,
data/raw/england_all_tiers.csv (jalapic/engsoccerdata on GitHub -- every
English league match since 1888, all four tiers), pairs every promoted club of
the Premier League era: 32 windows, 95 clubs (2022-23, which engsoccerdata
lacks, is filled from openfootball). That fit is the one the
pipeline uses when the file is present; the small one is kept for comparison.
football-data.co.uk, which also carries bookmaker closing odds, is ISP-blocked
from Indonesia and is not used.

Two inputs are fitted on the large sample: points per game and goal difference
per game (the latter is less noisy over a 46-match season). Whichever the
pipeline applies, both are printed with their n and r.

If b comes out near zero the line collapses to the population mean and nothing
changes -- the data decides how much the record is worth. Nothing here is
assumed: b, a, n and the correlation are all printed, and a club with no
Championship record on file keeps the population baseline.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import config

_FILE = config.DATA / "championship_records.json"
_ENG = config.RAW / "england_all_tiers.csv"
_PL_ERA_START = 1992           # first Premier League season, as engsoccerdata's start-year

# engsoccerdata skips 2022-23 entirely (no rows for Season 2022 in any tier).
# Filled from openfootball/football.json, same GitHub host, regular season only.
_GAP_FILL = {2022: ("openfootball_2022-23_en.1.json", "openfootball_2022-23_en.2.json")}


def load() -> dict:
    if not _FILE.exists():
        return {}
    return json.loads(_FILE.read_text(encoding="utf-8"))


def _pl_ppg(matches: pd.DataFrame, team: str, season: str) -> float | None:
    d = matches[(matches["season"] == season) &
                ((matches["home_team"] == team) | (matches["away_team"] == team))]
    if d.empty:
        return None
    pts = 0
    for r in d.itertuples(index=False):
        home = r.home_team == team
        gf, ga = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
        pts += config.POINTS_WIN if gf > ga else (config.POINTS_DRAW if gf == ga else config.POINTS_LOSS)
    return pts / len(d)


def _next_season(season: str) -> str:
    a, b = (int(x) for x in season.split("-"))
    return f"{a + 1}-{b + 1}"


def pairs(matches: pd.DataFrame, train_seasons: list[str] | None = None) -> pd.DataFrame:
    """
    (club, champ_season, champ_ppg, pl_ppg) for every past promoted club with
    both sides available. train_seasons restricts the PL side to seasons the
    caller is allowed to see -- the backtest passes the seasons before its fold.
    """
    data = load()
    allowed = set(train_seasons if train_seasons is not None else config.TRAIN_SEASONS)
    rows = []
    for champ_season, clubs in data.get("past", {}).items():
        pl_season = _next_season(champ_season)
        if pl_season not in allowed:
            continue
        for club, rec in clubs.items():
            if rec.get("ppg") is None:
                continue
            pl = _pl_ppg(matches, club, pl_season)
            if pl is None:
                continue
            rows.append({"club": club, "champ_season": champ_season,
                         "champ_ppg": float(rec["ppg"]), "pl_ppg": pl})
    return pd.DataFrame(rows)


def fit(matches: pd.DataFrame, train_seasons: list[str] | None = None) -> dict | None:
    """
    pl_ppg = a + b * champ_ppg on the past promoted clubs. None when fewer than
    three pairs exist -- a line through two points is not a measurement.

    b is clamped to [0, 1]: a negative slope would say a stronger Championship
    record predicts a WORSE Premier League one, and a slope above one that it
    predicts a bigger PL margin than it was itself -- both are noise, not
    football. The clamp is reported, not hidden.
    """
    p = pairs(matches, train_seasons)
    if len(p) < 3:
        return None
    x, y = p["champ_ppg"].to_numpy(), p["pl_ppg"].to_numpy()
    if x.std() < 1e-9:
        return None
    b_raw, a_raw = np.polyfit(x, y, 1)
    b = float(np.clip(b_raw, 0.0, 1.0))
    a = float(y.mean() - b * x.mean())        # re-anchor the intercept after clamping
    r = float(np.corrcoef(x, y)[0, 1])
    return {"a": a, "b": b, "b_raw": float(b_raw), "r": r, "n": len(p),
            "champ_mean": float(x.mean()), "pl_mean": float(y.mean()), "pairs": p}


def _eng() -> pd.DataFrame | None:
    if not _ENG.exists():
        return None
    e = pd.read_csv(_ENG, low_memory=False, usecols=["Season", "home", "visitor", "hgoal", "vgoal", "tier"])
    e = e[e["tier"].isin([1, 2])]
    for season, files in _GAP_FILL.items():
        if season not in set(e["Season"]):
            e = pd.concat([e, _openfootball(season, files, set(e["home"]))], ignore_index=True)
    return e


def _openfootball(season: int, files: tuple[str, ...], known: set[str]) -> pd.DataFrame:
    """One engsoccerdata-shaped frame for a season engsoccerdata lacks. Club
    names are openfootball's with the trailing FC/AFC dropped; any name that
    does not already exist in the file is an error, not a new club."""
    rows = []
    for tier, name in enumerate(files, start=1):
        path = config.RAW / name
        if not path.exists():
            raise FileNotFoundError(f"{path} -- engsoccerdata has no {season}-{season+1}; fetch "
                                    f"raw.githubusercontent.com/openfootball/football.json/master/"
                                    f"{season}-{str(season+1)[2:]}/en.{tier}.json")
        for m in json.load(open(path, encoding="utf-8"))["matches"]:
            if m.get("stage", "Regular") != "Regular" or not m.get("score", {}).get("ft"):
                continue
            h, v = (re.sub(r" A?FC$", "", m[t]) for t in ("team1", "team2"))
            rows.append({"Season": season, "home": h, "visitor": v,
                         "hgoal": m["score"]["ft"][0], "vgoal": m["score"]["ft"][1], "tier": tier})
    out = pd.DataFrame(rows)
    unknown = set(out["home"]) - known
    if unknown:
        raise ValueError(f"openfootball {season}: names not in engsoccerdata: {sorted(unknown)}")
    counts = out.groupby("tier").size().to_dict()
    if counts != {1: 380, 2: 552}:
        raise ValueError(f"openfootball {season}: expected 380/552 regular-season matches, got {counts}")
    return out


def _season_stats(e: pd.DataFrame, season: int, tier: int) -> pd.DataFrame:
    """club, games, ppg, gd_per_game for one (season, tier) in engsoccerdata."""
    d = e[(e["Season"] == season) & (e["tier"] == tier)]
    if d.empty:
        return pd.DataFrame(columns=["club", "games", "ppg", "gd"])
    home = pd.DataFrame({
        "club": d["home"], "gf": d["hgoal"], "ga": d["vgoal"],
        "pts": np.where(d["hgoal"] > d["vgoal"], config.POINTS_WIN,
                        np.where(d["hgoal"] == d["vgoal"], config.POINTS_DRAW, config.POINTS_LOSS)),
    })
    away = pd.DataFrame({
        "club": d["visitor"], "gf": d["vgoal"], "ga": d["hgoal"],
        "pts": np.where(d["vgoal"] > d["hgoal"], config.POINTS_WIN,
                        np.where(d["hgoal"] == d["vgoal"], config.POINTS_DRAW, config.POINTS_LOSS)),
    })
    both = pd.concat([home, away], ignore_index=True)
    g = both.groupby("club").agg(games=("pts", "size"), pts=("pts", "sum"),
                                 gf=("gf", "sum"), ga=("ga", "sum")).reset_index()
    g["ppg"] = g["pts"] / g["games"]
    g["gd"] = (g["gf"] - g["ga"]) / g["games"]
    return g[["club", "games", "ppg", "gd"]]


def big_pairs(min_season: int = _PL_ERA_START, before_pl_season: int | None = None) -> pd.DataFrame:
    """
    Every club in tier 2 in season N and tier 1 in season N+1, over the Premier
    League era: club, season, champ_ppg, champ_gd, pl_ppg, pl_gd. Names need no
    mapping -- a club is the same string in both tiers of the same file.

    before_pl_season: only windows whose PL side is < this start-year, so the
    backtest cannot see its own fold's promoted clubs in the fit.
    """
    e = _eng()
    if e is None:
        return pd.DataFrame()
    rows = []
    last = int(e["Season"].max())
    for n in range(min_season, last):
        if before_pl_season is not None and n + 1 >= before_pl_season:
            continue
        t2 = _season_stats(e, n, 2).set_index("club")
        t1 = _season_stats(e, n + 1, 1).set_index("club")
        for club in t2.index.intersection(t1.index):
            rows.append({"club": club, "season": n,
                         "champ_ppg": t2.at[club, "ppg"], "champ_gd": t2.at[club, "gd"],
                         "pl_ppg": t1.at[club, "ppg"], "pl_gd": t1.at[club, "gd"]})
    return pd.DataFrame(rows)


def fit_big(input_col: str = "gd", min_season: int = _PL_ERA_START,
            before_pl_season: int | None = None) -> dict | None:
    """
    pl_ppg = a + b * champ_<input> on every PL-era promoted club. input_col is
    "gd" (goal difference per game, less noisy over 46 matches) or "ppg". Same
    clamp to [0, 1] on b, same reporting, same refusal below three pairs.
    """
    p = big_pairs(min_season, before_pl_season)
    if len(p) < 3:
        return None
    x, y = p[f"champ_{input_col}"].to_numpy(), p["pl_ppg"].to_numpy()
    if x.std() < 1e-9:
        return None
    b_raw, _ = np.polyfit(x, y, 1)
    b = float(np.clip(b_raw, 0.0, 1.0))
    a = float(y.mean() - b * x.mean())
    r = float(np.corrcoef(x, y)[0, 1])
    return {"a": a, "b": b, "b_raw": float(b_raw), "r": r, "n": len(p),
            "champ_mean": float(x.mean()), "pl_mean": float(y.mean()),
            "input": input_col, "pairs": p}


def current_clubs() -> dict[str, dict]:
    """{club: {"ppg", "gd"}} for this season's promoted clubs, from the records file."""
    ppg, gd = current_ppg(), current_gd()
    return {c: {"ppg": ppg.get(c), "gd": gd.get(c)} for c in set(ppg) | set(gd)}


def current_gd() -> dict[str, float]:
    """{club: champ_gd_per_game} for this season's promoted clubs, where filled in."""
    data = load()
    out = {}
    for _, clubs in data.get("current", {}).items():
        for club, rec in clubs.items():
            if rec.get("gd") is not None:
                out[club] = float(rec["gd"])
    return out


def current_ppg() -> dict[str, float]:
    """{club: champ_ppg} for this season's promoted clubs, where filled in."""
    data = load()
    out = {}
    for _, clubs in data.get("current", {}).items():
        for club, rec in clubs.items():
            if rec.get("ppg") is not None:
                out[club] = float(rec["ppg"])
    return out


def past_ppg(champ_season: str) -> dict[str, float]:
    """{club: champ_ppg} for one past promotion season -- the backtest's 'current'."""
    data = load()
    return {club: float(rec["ppg"])
            for club, rec in data.get("past", {}).get(champ_season, {}).items()
            if rec.get("ppg") is not None}


def eng_champ_stats(season: int) -> dict[str, dict]:
    """{club: {"ppg", "gd"}} for tier 2 in one engsoccerdata season -- the
    backtest reads a fold's promoted clubs from here, same source as the fit."""
    e = _eng()
    if e is None:
        return {}
    t2 = _season_stats(e, season, 2)
    return {r.club: {"ppg": float(r.ppg), "gd": float(r.gd)} for r in t2.itertuples(index=False)}


def choose(matches: pd.DataFrame, clubs: dict[str, dict], train_seasons: list[str] | None = None,
           before_pl_season: int | None = None) -> tuple[dict | None, dict[str, float], str]:
    """
    Pick the offset model to apply and the matching per-club input.

    Preference: the PL-era fit on goal difference (largest n, best r), if every
    club in `clubs` has a gd; else the same fit on points per game; else the
    hand-collected small fit. Returns (model, {club: input_value}, label) so
    the caller can print which one it is applying and why.

    clubs: {club: {"ppg": ..., "gd": ...}} -- gd may be missing.
    """
    have_gd = clubs and all("gd" in v and v["gd"] is not None for v in clubs.values())
    have_ppg = clubs and all("ppg" in v and v["ppg"] is not None for v in clubs.values())

    if have_gd:
        m = fit_big("gd", before_pl_season=before_pl_season)
        if m:
            return m, {c: v["gd"] for c, v in clubs.items()}, f"PL-era GD/game (n={m['n']})"
    if have_ppg:
        m = fit_big("ppg", before_pl_season=before_pl_season)
        if m:
            return m, {c: v["ppg"] for c, v in clubs.items()}, f"PL-era PPG (n={m['n']})"
    m = fit(matches, train_seasons)
    if m and have_ppg:
        return m, {c: v["ppg"] for c, v in clubs.items()}, f"hand-collected PPG (n={m['n']})"
    return None, {}, "none"


def club_offset_ppg(club: str, model: dict | None, champ: dict[str, float]) -> float:
    """
    How far this club's own Championship record moves it from the population
    mean, in PL points-per-game. Zero when there is no fit or no record -- that
    club stays on promoted_baseline exactly as before.
    """
    if model is None or club not in champ:
        return 0.0
    return model["b"] * (champ[club] - model["champ_mean"])


def demo() -> None:
    # --- the arithmetic, on synthetic pairs -----------------------------------
    import tempfile, pathlib
    global _FILE
    real = _FILE
    try:
        tmp = pathlib.Path(tempfile.mkdtemp()) / "champ.json"
        _FILE = tmp
        # Known slope: PL ppg = 0.5 * champ ppg for three clubs, built from
        # four results each (D,D,D,L = 0.75; D,D,D,D = 1.00; W,D,D,L = 1.25).
        tmp.write_text(json.dumps({
            "past": {"2023-2024": {"A": {"ppg": 1.5}, "B": {"ppg": 2.0}, "C": {"ppg": 2.5}}},
            "current": {"2025-2026": {"X": {"ppg": 2.2}}},
        }), encoding="utf-8")
        results = {"A": [(1, 1), (1, 1), (1, 1), (0, 1)],
                   "B": [(1, 1), (1, 1), (1, 1), (1, 1)],
                   "C": [(2, 0), (1, 1), (1, 1), (0, 1)]}
        rows = [{"season": "2024-2025", "home_team": t, "away_team": "Opp",
                 "home_goals": g, "away_goals": a}
                for t, gs in results.items() for g, a in gs]
        m = pd.DataFrame(rows)
        got = fit(m, ["2024-2025"])
        assert got is not None and got["n"] == 3, got
        assert abs(got["b"] - 0.5) < 1e-9, f"slope should recover 0.5, got {got['b']}"
        assert abs(got["r"] - 1.0) < 1e-9, "a perfect line must have r = 1"
        x = current_ppg()
        assert abs(club_offset_ppg("X", got, x) - 0.5 * (2.2 - 2.0)) < 1e-9
        assert club_offset_ppg("nobody", got, x) == 0.0, "unknown club must be a no-op"
        # Too few pairs -> no fit, not a fabricated one.
        tmp.write_text(json.dumps({"past": {"2023-2024": {"A": {"ppg": 2.0}}}, "current": {}}),
                       encoding="utf-8")
        assert fit(m, ["2024-2025"]) is None
    finally:
        _FILE = real

    # --- what the real file says ---------------------------------------------
    from src.run_pipeline import load_matches
    matches = load_matches()
    model = fit(matches)
    champ = current_ppg()
    filled = len(pairs(matches))
    print(f"championship_records.json: {filled} past pairs filled, "
          f"{len(champ)} current clubs filled")
    if model is None:
        print("  no fit yet (need >= 3 past pairs) -- promoted clubs stay on the population baseline")
    else:
        print(f"  pl_ppg = {model['a']:.3f} + {model['b']:.3f} * champ_ppg   "
              f"(n={model['n']}, r={model['r']:.2f}, raw slope {model['b_raw']:.3f})")
        for club, cp in champ.items():
            off = club_offset_ppg(club, model, champ)
            print(f"  {club:16s} champ {cp:.2f} ppg -> {off:+.3f} ppg vs the population mean")
    print("OK  championship offset arithmetic checks pass")


if __name__ == "__main__":
    demo()
