"""
Historical manager assignments, for backtesting the manager layer.

    python -m src.manager_history        # structural check + what still needs verifying

data/manager_history.json says who managed each club at the start of the two
backtest seasons (2024-25, 2025-26), their status, and their prior stints. The
backtest reads it to apply run_pipeline.manager_delta() to the held-out season
instead of leaving the manager layer at its ESTIMATE default.

This module only loads and validates. It does not compute priors -- a stint
with a `prior_ref` reuses one from manager_priors.json, a stint with a
Premier League `league` is computable straight from matches_football_data.csv,
and anything else needs collection before the backtest can use that club.
"""

from __future__ import annotations

import json

import config

_FILE = config.DATA / "manager_history.json"
_STATUSES = {"new", "established", "interim_to_full"}


def load() -> dict:
    if not _FILE.exists():
        return {}
    return json.loads(_FILE.read_text(encoding="utf-8"))


def season(year: str) -> dict[str, dict]:
    """{club: manager record} for a backtest season, e.g. '2024-2025'."""
    return load().get("seasons", {}).get(year, {})


def is_verified() -> bool:
    return bool(load().get("meta", {}).get("verified"))


def _pl_stint_stats(matches, club: str, start: int, end: int,
                    train_seasons: list[str]) -> dict | None:
    """
    ppg for a Premier League stint, computed from matches_football_data.csv.

    Restricted to `train_seasons` -- the seasons strictly before the fold --
    so a 2025-26 backtest can never see 2025-26 data in a manager's prior. A
    season "A-B" counts for a stint [start, end] when start <= A and B <= end.
    The file has no xG, so these stints are ppg-only; manager_prior treats
    missing xG as league-average, which is the honest no-shape-signal default.
    """
    import config as _cfg

    seasons = []
    for s in train_seasons:
        a, b = (int(x) for x in s.split("-"))
        if start <= a and b <= end:
            seasons.append(s)
    if not seasons:
        return None

    pts = games = 0
    for s in seasons:
        d = matches[(matches["season"] == s) &
                    ((matches["home_team"] == club) | (matches["away_team"] == club))]
        for r in d.itertuples(index=False):
            home = r.home_team == club
            gf, ga = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
            pts += _cfg.POINTS_WIN if gf > ga else (_cfg.POINTS_DRAW if gf == ga else _cfg.POINTS_LOSS)
            games += 1
    if games == 0:
        return None
    return {"matches": games, "ppg": round(pts / games, 3),
            "source": f"matches_football_data.csv {'+'.join(seasons)}"}


def priors_for_season(year: str, matches=None, train_seasons: list[str] | None = None) -> tuple[dict, dict]:
    """
    (stint_stats, prior_stints) for a backtest season, both keyed by club, in
    the shape features.manager_prior expects.

    stint_stats[club][stint_club] = {matches, ppg, xg_seasons?}
    prior_stints[club] = [ {club, league, start, end}, ... ]

    A stint is resolved from, in order: an inline `prior` block, a `prior_ref`
    into manager_priors.json, or -- for a Premier League stint -- computed from
    the match file. Unresolved stints are dropped; a club with no resolved
    stint contributes no manager delta (same as an established manager).
    """
    import json as _json

    if train_seasons is None:
        # Default: everything before this fold's year, alphabetical order works
        # for "YYYY-YYYY" season strings.
        train_seasons = sorted(s for s in config.TRAIN_SEASONS if s < year)

    ref_file = config.MANAGER_PRIORS_FILE
    refs = _json.loads(ref_file.read_text(encoding="utf-8"))["priors"] if ref_file.exists() else {}
    flat_refs = {sc: st for club in refs.values() for sc, st in club.items()}

    stint_stats: dict[str, dict] = {}
    prior_stints: dict[str, list] = {}

    for club, rec in season(year).items():
        if rec.get("status") not in ("new", "interim_to_full"):
            continue
        resolved, kept = {}, []
        for st in rec.get("prior_stints", []):
            sc = st["club"]
            stats = None
            if st.get("prior"):
                stats = {k: v for k, v in st["prior"].items() if k in ("matches", "ppg", "xg_seasons")}
            elif st.get("prior_ref") and st["prior_ref"] in flat_refs:
                r = flat_refs[st["prior_ref"]]
                stats = {k: r[k] for k in ("matches", "ppg", "xg_seasons") if k in r}
            elif st.get("league") == "Premier League" and matches is not None:
                stats = _pl_stint_stats(matches, sc, st["start"], st["end"], train_seasons)
            if stats and "ppg" in stats:
                resolved[sc] = stats
                kept.append({k: st[k] for k in ("club", "league", "start", "end") if k in st})
        if resolved:
            stint_stats[club] = resolved
            prior_stints[club] = kept

    return stint_stats, prior_stints


def _validate(data: dict) -> list[str]:
    """Structural problems and coverage gaps, as plain strings."""
    problems = []
    seasons = data.get("seasons", {})

    for year in config.BACKTEST_SEASONS:
        clubs = seasons.get(year)
        if not clubs:
            problems.append(f"{year}: no clubs listed")
            continue
        if len(clubs) != 20:
            problems.append(f"{year}: {len(clubs)} clubs, expected 20")

        for club, rec in clubs.items():
            if rec.get("status") not in _STATUSES:
                problems.append(f"{year}/{club}: status {rec.get('status')!r} not in {_STATUSES}")
            if not rec.get("manager"):
                problems.append(f"{year}/{club}: no manager name")
            if rec.get("status") in ("new", "interim_to_full") and not rec.get("prior_stints"):
                problems.append(f"{year}/{club}: {rec.get('status')} manager with no prior_stints")

    return problems


def _stint_readiness(data: dict) -> dict[str, list[str]]:
    """Which new-manager stints are ready to use, and which need collection."""
    ready, needs = [], []
    for year, clubs in data.get("seasons", {}).items():
        for club, rec in clubs.items():
            if rec.get("status") not in ("new", "interim_to_full"):
                continue
            for st in rec.get("prior_stints", []):
                tag = f"{year}/{club} <- {st['club']}"
                if st.get("prior") or st.get("prior_ref") or st.get("league") == "Premier League":
                    ready.append(tag)
                else:
                    needs.append(f"{tag} ({st.get('league', '?')})")
    return {"ready": ready, "needs_collection": needs}


def demo() -> None:
    data = load()
    assert data, "data/manager_history.json missing"

    problems = _validate(data)
    if problems:
        print(f"{len(problems)} structural problem(s):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("structure OK: both backtest seasons, 20 clubs each, statuses valid")

    r = _stint_readiness(data)
    # A manager is usable once ANY one stint has data -- the prior chain
    # weights by recency and sample, so the primary stint dominates and a
    # missing older/secondary one only slightly narrows the evidence.
    usable, blocked = [], []
    for year, clubs in data.get("seasons", {}).items():
        for club, rec in clubs.items():
            if rec.get("status") not in ("new", "interim_to_full"):
                continue
            stints = rec.get("prior_stints", [])
            has = any(st.get("prior") or st.get("prior_ref")
                      or st.get("league") == "Premier League" for st in stints)
            (usable if has or not stints else blocked).append(f"{year}/{club}")

    print(f"\n{len(usable)} new-manager clubs usable (>=1 stint has data):")
    print("  " + ", ".join(usable))
    if blocked:
        print(f"{len(blocked)} still blocked (no stint has data): {', '.join(blocked)}")
    print(f"\n{len(r['needs_collection'])} secondary/older stints missing data (optional -- "
          "primary stint carries the prior):")
    for n in r["needs_collection"]:
        print(f"  - {n}")

    low = [f"{y}/{c}" for y, clubs in data["seasons"].items()
           for c, rec in clubs.items() if rec.get("confidence") == "low"]
    print(f"\n{len(low)} entries flagged confidence 'low' -- verify these first:")
    for e in low:
        print(f"  - {e}")

    print(f"\nmeta.verified = {is_verified()}"
          + ("" if is_verified() else "  -- backtest --with-manager stays off until this is true"))
    assert not problems, "fix the structural problems above"


if __name__ == "__main__":
    demo()
