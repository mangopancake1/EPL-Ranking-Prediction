"""
Squad quality from real Premier League player ratings (fotmob_ratings.md,
10 seasons, user-sourced).

    python -m src.player_ratings

Why this exists: a fitted club's attack/defence comes entirely from its own
match results, so it knows nothing about a squad that has since changed. Sign
an EPL-proven player with a 7.0 rating and the fit does not move -- his
quality was earned elsewhere and the fit never saw him in these colours.
`adaptation_factor()` cannot help either: anyone past EPL_PROVEN_MATCHES
returns a flat 1.0, so 191 of the 571 squad players carry no quality signal
at all.

The fix is NOT to add a squad bonus on top of the fit -- a player who was
already at the club is inside the fitted strength, and adding his rating
again would count him twice. Instead this supplies a SECOND estimate of the
same quantity, from the current squad, which run_pipeline blends with the
fitted one (config.SQUAD_RATING_BLEND). A player present in both estimates is
fine; that is what blending means. A newly-arrived player appears only in the
squad estimate, and pulls the blend toward his new club -- which is exactly
the missing information.

The rating is a league-level measure over a player's total matches there, not
a club attribute, so it travels with the player between clubs.

The rating -> strength slope is MEASURED at runtime by regressing the fitted
clubs' own combined strength on their squad rating, the same way
run_pipeline.ppg_to_strength_slope measures its own conversion. Only the
blend weight is an estimate.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

import config

_SEASON = re.compile(r"## (\d{4})/(\d{4})\n")
_ROW = re.compile(r"^\| *\d+ *\| *([^|]+?) *\| *\d+ *\| *([\d.]+) *\|$", re.M)


@lru_cache(maxsize=1)
def parse(path=None) -> dict[str, dict[int, float]]:
    """{player: {season_end_year: rating}} from fotmob_ratings.md."""
    path = path or (config.SCRAPED / "fotmob_ratings.md")
    if not path.exists():
        return {}
    parts = _SEASON.split(path.read_text(encoding="utf-8"))
    out: dict[str, dict[int, float]] = defaultdict(dict)
    # [preamble, y1, y2, body, y1, y2, body, ...]
    for i in range(1, len(parts), 3):
        end, body = int(parts[i + 1]), parts[i + 2]
        for name, rating in _ROW.findall(body):
            out[name.strip()][end] = float(rating)
    return dict(out)


def player_rating(name: str, as_of_year: int) -> float | None:
    """
    One player's Premier League rating, recency-weighted across the seasons
    he appears in. None when he isn't in the file at all -- an unknown
    rating, not a bad one, and callers drop him rather than assume a value.

    Same half-life as debutant_career uses, one philosophy for staleness.
    """
    seasons = parse().get(name)
    if not seasons:
        return None
    num = den = 0.0
    for end, rating in seasons.items():
        w = math.exp(-math.log(2) * max(0, as_of_year - end) / config.PLAYER_RATING_HALFLIFE_YEARS)
        num += w * rating
        den += w
    return num / den if den > 0 else None


def current_squad(players: pd.DataFrame, club: str) -> pd.DataFrame:
    """
    One club's squad list with this summer's real transfers applied.

    Departed players are dropped and priced arrivals appended. Only arrivals
    the squad list doesn't already carry: most summer signings ARE already in
    it, since the squad paste post-dates them, and concatenating them again
    would count one player twice -- double weight in the value-weighted mean,
    and two of the eighteen slots.

    Shared deliberately. squad_rating() applied this and the dashboard did not,
    so the two disagreed about who plays for whom: seven departed players still
    appeared on the Compare page while seven priced arrivals were missing from
    it. One helper means they cannot drift apart again.
    """
    out, in_ = roster_delta(club)
    kept = players[~players["player"].isin(out)]
    already = set(kept["player"])
    # A name on both lists was signed and then loaned straight back out --
    # Brughmans to Liverpool and on to Genk the same day, Orozco, Satpaev, van
    # Oevelen, Gustavo Sa. The departure is later in every case, so he is not
    # available this season and must not be re-added as an arrival.
    fresh = [p for p in in_ if p["player"] not in already and p["player"] not in out]
    if not fresh:
        return kept

    # An arrival the squad paste never saw knows only his name, price and
    # position. Everything else is genuinely absent, and absence has a correct
    # value here: a player who has not appeared for this club has no Premier
    # League matches and no minutes. Left as NaN they would crash the callers
    # that cast to int, or silently fall outside every position line -- the
    # same failure mode as the position-vocabulary bug.
    from src.build_squad_values import _POSITION_ALIASES

    incoming = pd.DataFrame(fresh).reindex(columns=kept.columns)
    incoming["position"] = [
        _POSITION_ALIASES.get(p.get("position") or "", p.get("position")) for p in fresh
    ]
    for column, default in (("epl_matches", 0), ("minutes_last_season", 0),
                            ("source_league", "Unknown"), ("fpl_matched", False)):
        if column in incoming.columns:
            incoming[column] = default
    return pd.concat([kept, incoming], ignore_index=True)


def squad_rating(players: pd.DataFrame, as_of_year: int,
                 club: str | None = None) -> tuple[float | None, int]:
    """
    A club's squad rating: the value-weighted mean rating of its top-N by
    market value, plus how many of those N actually had a rating.

    Weighted by value rather than headcount for the same reason
    squad_strength() is -- the players who decide matches are not a flat
    average of the squad list. Returns (None, n) when too few are covered to
    say anything honest; fotmob's table is a top-N per season, so a squad of
    fringe players legitimately has almost no rated names.

    club, if given, applies this summer's real transfers on top of the
    squad list before ranking it (see roster_delta()) -- without this, a
    club that sold or loaned out a starter keeps his rating in its top-18
    for as long as the underlying squad snapshot goes unrefreshed, and a
    club's new signing who arrived after that snapshot was taken earns
    nothing at all.
    """
    if club:
        players = current_squad(players, club)

    # A player with no Premier League rating is not necessarily unmeasured:
    # 117 of this season's arrivals have a real senior career elsewhere, and
    # debutant_career already expresses it in Premier League terms. Falling
    # back to that beats dropping them, which was leaving Hull City with 1
    # rated player of 18 and Coventry with 4 -- both under the minimum, so the
    # two clubs with the least match evidence got no squad rating at all.
    # Preference order matters: a real EPL rating always wins over an estimate
    # translated from another league.
    from src import debutant_career

    career = debutant_career.rating_lookup()
    top = players.nlargest(config.SQUAD_DEPTH_COUNT, "market_value_m").copy()
    top["rating"] = [
        epl if (epl := player_rating(p, as_of_year)) is not None else career.get(p)
        for p in top["player"]
    ]
    rated = top.dropna(subset=["rating"])
    if len(rated) < config.SQUAD_RATING_MIN_PLAYERS or rated["market_value_m"].sum() <= 0:
        return None, len(rated)
    value = rated["market_value_m"]
    return float((rated["rating"] * value).sum() / value.sum()), len(rated)


# Transfer-file club labels (transfers_2026_summer.md, user-sourced) -> the
# canonical config.TEAMS name. The file uses full Transfermarkt names both in
# its section headings and in the counterparty column; only listed where they
# differ from ours.
_TRANSFER_ALIASES = {
    "AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal",
    "Brentford FC": "Brentford", "Brighton & Hove Albion": "Brighton",
    "Chelsea FC": "Chelsea", "Everton FC": "Everton", "Fulham FC": "Fulham",
    "Liverpool FC": "Liverpool", "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd", "Nottingham Forest": "Nott'ham Forest",
    "Sunderland AFC": "Sunderland", "Tottenham Hotspur": "Tottenham",
}

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)
_DIRECTION_RE = re.compile(r"^\*\*(IN|OUT)\b", re.M)
_ROW_RE = re.compile(
    r"^\| *(.+?) *\| *(.+?) *\| *(.+?) *\| *(.+?) *\| *(.+?) *\| *(.+?) *\|$", re.M)
_FEE_RE = re.compile(r"€\s*([\d.]+)\s*([MK])", re.IGNORECASE)


def _market_value_m(mv: str) -> float | None:
    """
    The player's market value at the time of the move, in €m, or None when the
    file records it as unknown ("?").

    This deliberately reads the MV column, not the fee column. A fee answers
    "what did this move cost", which for a loan is a season's rent (Kevin
    Danso at 2.9m) and for a free transfer is nothing at all -- neither is a
    measure of the player. MV answers "what is he worth", which is what a
    value-weighted squad rating needs, and it exists for loans and frees too.
    """
    m = _FEE_RE.search(mv)
    if not m:
        return None
    num = float(m.group(1))
    return num / 1000.0 if m.group(2).upper() == "K" else num


@lru_cache(maxsize=1)
def _transfers() -> list[dict]:
    """
    Every move in the file, flattened to {"from", "to", "player", "value_m"}.

    The file is organised per club, as an IN block and an OUT block under a
    "## Club" heading, so direction comes from the block a row sits in rather
    than from the row itself. Both sides of a Premier-League-to-Premier-League
    move appear in the file (Konsa under Arsenal IN and under Aston Villa
    OUT), which is why no row needs to be mirrored here.
    """
    path = config.SCRAPED / "transfers_2026_summer.md"
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    # Split into (club, body) sections, then each section into IN / OUT blocks.
    marks = [(m.start(), m.group(1)) for m in _SECTION_RE.finditer(text)]
    rows = []
    for i, (pos, club_raw) in enumerate(marks):
        club = _TRANSFER_ALIASES.get(club_raw, club_raw)
        body = text[pos:marks[i + 1][0] if i + 1 < len(marks) else len(text)]

        blocks = [(m.start(), m.group(1)) for m in _DIRECTION_RE.finditer(body)]
        for j, (bpos, direction) in enumerate(blocks):
            chunk = body[bpos:blocks[j + 1][0] if j + 1 < len(blocks) else len(body)]
            for name, pos, other_raw, _date, mv, _fee in _ROW_RE.findall(chunk):
                if name == "Pemain" or set(name) <= {"-"}:
                    continue  # header row / separator
                other = _TRANSFER_ALIASES.get(other_raw, other_raw)
                frm, to = (other, club) if direction == "IN" else (club, other)
                rows.append({"from": frm, "to": to, "player": name,
                             "position": pos, "value_m": _market_value_m(mv)})
    return rows


def roster_delta(club: str) -> tuple[set[str], list[dict]]:
    """
    (departed, arrived) for one club, from this summer's real transfers.

    departed: names who left `club` this window (sold, loaned out, or
    released) -- excluded from its squad rating regardless of whether the
    underlying squad snapshot has caught up yet.

    arrived: [{"player", "market_value_m"}] for names who joined `club` from
    outside it, priced off the transfer fee when one is public. A loan/free
    move with no disclosed fee is still recorded as a departure/arrival of a
    PLAYER, just with no value to weight him by, so he cannot enter a squad
    rating computed by value -- an honest gap, not a fabricated price.
    """
    departed, arrived = set(), []
    for t in _transfers():
        if t["from"] == club and t["to"] != club:
            departed.add(t["player"])
        elif t["to"] == club and t["from"] != club and t["value_m"] is not None:
            arrived.append({"player": t["player"], "market_value_m": t["value_m"],
                            "position": t.get("position")})
    # The transfer file is appended to by hand as windows progress, so the same
    # move can be pasted twice. Keep the first row for a given player.
    seen, unique = set(), []
    for a in arrived:
        if a["player"] not in seen:
            seen.add(a["player"])
            unique.append(a)
    return departed, unique


def implied_strength(squad_ratings: dict[str, float],
                     fitted_combined: dict[str, float]) -> tuple[dict[str, float], dict]:
    """
    Convert each club's squad rating into combined-strength units, using a
    slope measured from the fitted clubs themselves.

    Regressing the fit on the squad rating is deliberately circular-looking
    and that is the point: it asks "in this league, this season, what is one
    rating point worth in strength?" and answers it from real numbers rather
    than a chosen constant. Returns the per-club implied strength plus the
    regression's own diagnostics, so a weak fit can be seen rather than
    silently trusted.
    """
    # Fitted on the clubs that have both -- a promoted club has no fitted
    # strength to regress against -- but SCORED for every club with a squad
    # rating, so a promoted side can be read off the same line the rest of the
    # league defines. That is an extrapolation only if its rating sits outside
    # the fitted range; the diagnostics report the range so it can be seen.
    teams = [t for t in squad_ratings if t in fitted_combined]
    x = np.array([squad_ratings[t] for t in teams])
    y = np.array([fitted_combined[t] for t in teams])
    if len(teams) < 3 or x.std() == 0:
        return {}, {"n": len(teams), "r": float("nan"), "slope": float("nan")}

    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    implied = {t: float(slope * q + intercept) for t, q in squad_ratings.items()}
    return implied, {"n": len(teams), "r": r, "r2": r ** 2,
                     "slope": float(slope), "intercept": float(intercept),
                     "fit_lo": float(x.min()), "fit_hi": float(x.max())}


def demo() -> None:
    import json

    ratings = parse()
    assert ratings, "expected fotmob_ratings.md to parse"
    assert all(5.0 < r < 10.0 for s in ratings.values() for r in s.values()), \
        "ratings outside a believable 5-10 band -- parser probably grabbed the wrong column"

    as_of = pd.Timestamp(config.TODAY).year
    # Recency weighting must actually favour the recent season.
    seasons = {2026: 8.0, 2020: 6.0}
    blended = sum(
        math.exp(-math.log(2) * (as_of - e) / config.PLAYER_RATING_HALFLIFE_YEARS) * r
        for e, r in seasons.items()
    ) / sum(
        math.exp(-math.log(2) * (as_of - e) / config.PLAYER_RATING_HALFLIFE_YEARS)
        for e in seasons
    )
    assert blended > 7.0, f"recent 8.0 should outweigh a stale 6.0, got {blended:.2f}"

    players = pd.DataFrame(json.loads(config.SQUAD_VALUES_FILE.read_text(encoding="utf-8"))["players"])
    covered = {}
    for team in config.TEAMS:
        q, n = squad_rating(players[players["club"] == team], as_of)
        if q is not None:
            covered[team] = q
    print(f"{len(covered)}/{len(config.TEAMS)} clubs have enough rated players")
    for t, q in sorted(covered.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {t:18s} squad rating {q:.3f}")

    # roster_delta must actually move a real, known transfer in each
    # direction, not just parse without crashing.
    out, in_ = roster_delta("Manchester City")
    assert "Rodri" in out, "Rodri -> Barcelona should read as a City departure"
    assert any(p["player"] == "Elliot Anderson" for p in in_), \
        "Elliot Anderson's priced arrival at City should be captured"
    print(f"roster_delta sanity check OK ({len(out)} out, {len(in_)} priced in, Man City)")

    # Most summer signings are ALREADY in the squad list -- the paste
    # post-dates them -- so the arrival merge has to dedupe or a player gets
    # counted twice: double weight in the value-weighted mean, and two of the
    # eighteen slots. Ipswich had nine such overlaps when this was caught.
    overlapping = 0
    for team in config.TEAMS:
        own = players[players["club"] == team]
        departed, arrived = roster_delta(team)
        kept = own[~own["player"].isin(departed)]
        overlap = {p["player"] for p in arrived} & set(kept["player"])
        overlapping += len(overlap)
        pool = list(kept["player"]) + [p["player"] for p in arrived if p["player"] not in set(kept["player"])]
        assert len(pool) == len(set(pool)), f"{team}: duplicate player in the squad-rating pool"
    assert overlapping > 0, "expected some signings to appear in both sources -- guard is untested otherwise"

    # The career-rating fallback: it has to lift the clubs it was added for,
    # and it must never override a real Premier League rating.
    from src import debutant_career

    career = debutant_career.rating_lookup()
    both = [n for n in career if player_rating(n, as_of) is not None]
    assert both, "expected at least one player with both an EPL rating and a career rating"
    epl_only = [n for n in parse() if n not in career]
    sample = both + epl_only[:config.SQUAD_RATING_MIN_PLAYERS - len(both)]
    frame = pd.DataFrame({"player": sample, "market_value_m": [10.0] * len(sample)})
    used, _ = squad_rating(frame, as_of)
    expected = sum(player_rating(n, as_of) for n in sample) / len(sample)
    assert used is not None and abs(used - expected) < 1e-9, (
        f"career rating overrode a real EPL one: got {used}, EPL-only mean is {expected}")
    for team, floor in (("Hull City", 10), ("Coventry City", 10)):
        _, n = squad_rating(players[players["club"] == team], as_of, club=team)
        assert n >= floor, f"{team}: only {n} rated -- career fallback not applied"
    covered = sum(
        1 for team in config.TEAMS
        if squad_rating(players[players["club"] == team], as_of, club=team)[0] is not None
    )
    assert covered == len(config.TEAMS), f"only {covered}/{len(config.TEAMS)} clubs rated"
    print(f"career-rating fallback OK ({len(career)} careers, all {covered} clubs now rated)")

    # implied_strength has to score every club it is given a rating for, not
    # only the ones it could fit on -- that is what lets a promoted club, which
    # has no fitted strength, be read off the league's own line.
    ratings = {t: squad_rating(players[players["club"] == t], as_of, club=t)[0]
               for t in config.TEAMS}
    ratings = {t: q for t, q in ratings.items() if q is not None}
    fitted_like = {t: 0.5 * (q - config.PLAYER_RATING_AVG) for t, q in ratings.items()
                   if t not in config.PROMOTED}
    implied, diag = implied_strength(ratings, fitted_like)
    assert set(implied) == set(ratings), "implied_strength dropped clubs it was asked to score"
    assert diag["n"] == len(fitted_like), "fit should use only the clubs with both values"
    assert diag["fit_lo"] <= diag["fit_hi"]
    print(f"implied_strength OK (fit on {diag['n']}, scored {len(implied)})")

    # A player signed and immediately loaned out belongs to neither squad.
    both = 0
    for team in config.TEAMS:
        gone, arrived = roster_delta(team)
        overlap = {a["player"] for a in arrived} & gone
        both += len(overlap)
        roster = set(current_squad(players[players["club"] == team], team)["player"])
        assert not (roster & overlap), f"{team}: {roster & overlap} left but still counted"
    assert both > 0, "expected some signed-then-loaned-out players -- guard is untested otherwise"
    print(f"in-and-out guard OK ({both} players signed then loaned straight back out)")
    print(f"dedupe check OK ({overlapping} signings present in both squad list and transfer list)")


if __name__ == "__main__":
    demo()
