"""
Real pre-EPL senior-career signal per debutant: a rating from
karier_debutan_epl.md (per-season history: club, matches, goals, assists,
rating) and the league it was earned in, from debutan_epl_2627.md's "Liga
Sebelumnya" column. Both user-sourced.

    python -m src.debutant_career

Feeds adaptation_factor() a genuine individual-quality signal instead of a
league-only guess, for the debutants whose name matches a squad entry.
Reduces each player's "Klub Senior" rows to one number, per
config.PLAYER_CONSISTENCY_PENALTY's rule: reward a reliably-good career over
one standout season among otherwise mediocre ones, not just the raw average.

The rating alone isn't the whole story, though: a great rating earned in a
weak league still needs a league penalty applied, same as it would for any
other player -- a shining spell in a league far below the Premier League
should not translate into full credit here. Mirrors how run_pipeline treats
a manager's prior stints exactly, season by season rather than one flat
career-wide league: each season's OWN club (club_leagues.CLUB_LEAGUE) sets
its league_weight, which discounts the rating twice, on purpose, answering
two different questions -- once on the size of the edge above average
(features._rescale, same as a manager's ppg/xG margin), and once on how
much that season's matches count toward being "proven" at all (the sample
term below, same as features.stint_weight's effective_matches). A season at
a club not in CLUB_LEAGUE (reserve/youth/lower-tier sides, mostly) falls
back to a neutral weight of 1.0 -- skipped rather than guessed, not
penalised for being unmapped. league_lookup() (a single career-wide guess
from debutan_epl_2627.md) is now only the fallback for a player with zero
per-season club matches, not the primary signal.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

import pandas as pd

import config
from src.club_leagues import CLUB_LEAGUE

_HEADER = re.compile(r"^## (.+)$", re.M)
_SENIOR_TABLE = re.compile(
    r"\*\*Klub Senior\*\*\s*\n\n\|.*?\|\n\|[-|]+\|\n((?:\|.*\|\n?)+)"
)
_ROW = re.compile(
    r"^\|\s*(.+?)\s*\|\s*(\d{4})/(\d{4})\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([\d.]*)\s*\|$",
    re.M,
)


def _parse_player_block(block: str) -> list[dict]:
    """Senior-club rows with a real rating, for one player's block of text."""
    m = _SENIOR_TABLE.search(block)
    if not m:
        return []
    rows = []
    for club, _y1, y2, played, rating in _ROW.findall(m.group(1)):
        if not rating:
            continue  # season played, but no rating recorded -- not usable
        rows.append({"club": club.strip(), "end": int(y2), "matches": int(played),
                     "rating": float(rating)})
    return rows


def parse(path=None) -> dict[str, list[dict]]:
    """{player_name: [{"end", "matches", "rating"}, ...]} for senior-career rows with a rating."""
    path = path or (config.SCRAPED / "karier_debutan_epl.md")
    text = path.read_text(encoding="utf-8")
    parts = _HEADER.split(text)[1:]  # [name1, body1, name2, body2, ...]
    out = {}
    for name, body in zip(parts[0::2], parts[1::2]):
        rows = _parse_player_block(body)
        if rows:
            out[name.strip()] = rows
    return out


def career_rating(rows: list[dict], as_of_year: int) -> float | None:
    """
    Consistency-weighted senior-career rating, on the real rating scale.

    Each season is weighted by recency (config.PLAYER_RATING_HALFLIFE_YEARS)
    and by how much of that season the player actually played
    (config.PLAYER_RATING_FULL_MATCHES) -- a two-game cameo can't outweigh a
    real season. None if nothing survives, rather than dividing by zero.

    This is a pre-Premier-League rating expressed in Premier League terms:
    each season's edge over the league-average rating is discounted by its own
    league's weight, so a 7.5 in the Eredivisie does not read as a 7.5 here.
    That discount is what makes it comparable enough to stand in for a missing
    EPL rating; it is still an estimate from another league, not a measurement
    of this one. Clamped to the same floor/ceiling the rest of the model uses,
    so one freak season cannot drag a squad mean.
    """
    if not rows:
        return None

    # Deferred: features.py imports this module too, at the top level.
    from src.features import league_weight

    weights, ratings = [], []
    for r in rows:
        lw = league_weight(CLUB_LEAGUE.get(r.get("club", ""), ""))
        if lw is None:
            lw = 1.0  # club not mapped: neutral, not penalised for being unmapped

        years_ago = max(0.0, as_of_year - r["end"])
        recency = math.exp(-math.log(2) * years_ago / config.PLAYER_RATING_HALFLIFE_YEARS)
        # Matches in a weaker league count less toward "proven", same
        # principle as features.stint_weight's effective_matches for managers.
        sample = min(r["matches"] * lw / config.PLAYER_RATING_FULL_MATCHES, 1.0)
        w = recency * sample
        if w > 0:
            weights.append(w)
            # Only the edge over the league-wide average rating transfers,
            # discounted by league strength -- same role features._rescale
            # plays for a manager's ppg/xG margin.
            ratings.append(config.PLAYER_RATING_AVG + (r["rating"] - config.PLAYER_RATING_AVG) * lw)

    if not weights:
        return None

    total = sum(weights)
    mean = sum(w * r for w, r in zip(weights, ratings)) / total

    if len(weights) > 1:
        variance = sum(w * (r - mean) ** 2 for w, r in zip(weights, ratings)) / total
        penalised = mean - config.PLAYER_CONSISTENCY_PENALTY * math.sqrt(variance)
    else:
        penalised = mean  # one season alone says nothing about consistency

    return float(min(max(penalised, config.PLAYER_RATING_FLOOR),
                     config.PLAYER_RATING_CEILING))


def score(rows: list[dict], as_of_year: int) -> float | None:
    """career_rating() rescaled to the 0..1 quality terms adaptation_factor() uses."""
    rating = career_rating(rows, as_of_year)
    if rating is None:
        return None
    span = config.PLAYER_RATING_CEILING - config.PLAYER_RATING_FLOOR
    return float(min(max((rating - config.PLAYER_RATING_FLOOR) / span, 0.0), 1.0))


@lru_cache(maxsize=1)
def quality_lookup() -> dict[str, float]:
    """{player_name: 0..1 quality} for every debutant with a usable senior-career rating."""
    as_of_year = pd.Timestamp(config.TODAY).year
    out = {}
    for name, rows in parse().items():
        q = score(rows, as_of_year)
        if q is not None:
            out[name] = q
    return out


@lru_cache(maxsize=1)
def rating_lookup() -> dict[str, float]:
    """{player_name: career rating} on the real scale, for squad_rating's fallback."""
    as_of_year = pd.Timestamp(config.TODAY).year
    out = {}
    for name, rows in parse().items():
        r = career_rating(rows, as_of_year)
        if r is not None:
            out[name] = r
    return out


# Free-text "Liga Sebelumnya" fragments (from debutan_epl_2627.md, sourced by
# the user) -> the exact league_weights.json key. A shining rating in a weak
# league still needs the league penalty applied -- this is what lets
# adaptation_factor() find the real league instead of falling back to the
# blind "Unknown" it would otherwise get (debutants have zero EPL matches, so
# build_squad_values.py's own source_league guess is always "Unknown").
# Best-effort substring match, ordered so a more specific phrase (e.g.
# "Bundesliga Austria") is tried before a more general one ("Bundesliga").
# Anything not covered here stays unmapped on purpose -- a wrong league is
# worse than an honest "don't know", which adaptation_factor already handles.
_LEAGUE_ALIASES = [
    ("Bundesliga Austria", "Austrian Bundesliga"),
    ("Bundesliga", "Bundesliga"),
    ("Ligue 1", "Ligue 1"),
    ("Ligue 2", "Ligue 2"),
    ("Primeira Liga", "Primeira Liga"),
    ("La Liga", "La Liga"),
    ("Primera RFEF", "La Liga"),  # Spain's 3rd tier isn't listed; La Liga's weight is the closest real anchor
    ("Eredivisie", "Eredivisie"),
    ("Serie A", "Serie A"),
    ("Serie B", "Serie A"),  # not listed separately; nearest real anchor
    ("EFL Championship", "Championship"),
    ("League One", "League One"),
    ("Swiss Super League", "Swiss Super League"),
    ("Super League, Swiss", "Swiss Super League"),
    ("Super League, Yunani", "Super League Greece"),
    ("Scottish Premiership", "Scottish Premiership"),
    ("Superliga, Denmark", "Danish Superliga"),
    ("HNL", "Croatian HNL"),
    ("Eliteserien", "Norwegian Eliteserien"),
    ("Pro League, Belgia", "Belgian Pro League"),
    ("Liga Argentina", "Argentine Primera"),
    ("River Plate", "Argentine Primera"),
    ("J-League", "J1 League"),
    ("Série A, Brasil", "Brazilian Serie A"),
    ("MLS", "MLS"),
]


@lru_cache(maxsize=1)
def league_lookup() -> dict[str, str]:
    """{player_name: league_weights.json key}, best-effort, from debutan_epl_2627.md."""
    path = config.SCRAPED / "debutan_epl_2627.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    rows = re.findall(r"^\| (.+?) \| .+? \| (.+?) \|$", text, re.M)
    out = {}
    for name, prev in rows:
        if name == "Pemain":
            continue
        for fragment, league in _LEAGUE_ALIASES:
            if fragment in prev:
                out[name.strip()] = league
                break
    return out


def demo() -> None:
    lut = quality_lookup()
    assert lut, "expected at least one debutant with a usable rating"
    assert all(0.0 <= v <= 1.0 for v in lut.values()), "quality must stay in 0..1"

    leagues = league_lookup()
    assert leagues, "expected at least one debutant mapped to a real league"
    from src.features import league_weight
    unresolved = [lg for lg in set(leagues.values()) if league_weight(lg) is None]
    assert not unresolved, f"league_lookup produced a key league_weights.json doesn't have: {unresolved}"

    # A reliably-solid career should outscore a one-big-season career at a
    # similar or lower average -- the whole point of the consistency penalty.
    steady = score([{"end": 2026, "matches": 30, "rating": 7.0},
                    {"end": 2025, "matches": 30, "rating": 7.0},
                    {"end": 2024, "matches": 30, "rating": 7.0}], 2026)
    spiky = score([{"end": 2026, "matches": 30, "rating": 8.5},
                   {"end": 2025, "matches": 30, "rating": 6.0},
                   {"end": 2024, "matches": 30, "rating": 6.2}], 2026)
    assert steady > spiky, f"consistency should win: steady={steady}, spiky={spiky}"

    print(f"{len(lut)} debutants with a usable pre-EPL rating")
    for name, q in sorted(lut.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {name:28s} quality={q:.2f}")


if __name__ == "__main__":
    demo()
