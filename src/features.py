"""
Feature layer: league weights, manager priors, player adaptation, fatigue.

This is where the design decisions live. Everything here converts raw scraped
numbers into something the Dixon-Coles fit or the simulator can consume.

Guiding rule, inherited from the project brief: a missing weight stays None
rather than becoming a plausible-looking guess. Downstream code widens the
uncertainty band instead of pretending to know.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

import config
from src import debutant_career


# ------------------------------------------------------------ league weights


@lru_cache(maxsize=1)
def load_league_weights() -> dict:
    with open(config.LEAGUE_WEIGHTS_FILE, encoding="utf-8") as fh:
        return json.load(fh)["leagues"]


def league_weight(league: str) -> float | None:
    """
    Strength of a league relative to the Premier League.

    Returns None for leagues with no defensible number -- international
    football, UEFA-suspended competitions, anything unlisted. Callers must
    handle None; substituting a default here would launder a guess into a
    fact.
    """
    entry = load_league_weights().get(league)
    if entry is None:
        return None
    return entry.get("weight")


def league_confidence(league: str) -> str:
    entry = load_league_weights().get(league)
    return entry.get("confidence", "none") if entry else "none"


# ------------------------------------------------------------ manager prior


def stint_weight(stint: dict, as_of: pd.Timestamp) -> float:
    """
    How much a past job should count. Four multiplicative discounts.

    League quality applies TWICE, answering two different questions: once
    here on the volume of experience (a season in the Austrian Bundesliga
    should not build toward "fully proven" as fast as a season in La Liga
    does -- top-league time is worth more per match, not just per point),
    and again in manager_prior() on the size of the edge itself (whatever
    that experience was worth, dominating a weak league transfers less of
    it to the Premier League). Being multiplied in twice is deliberate, not
    a double-count -- "how much do I trust this stint" and "how much does
    its edge translate" are genuinely separate questions.

    The other two: how long ago it ended, and how much of it there was. A
    brilliant half-season in the Austrian Bundesliga eight years ago should
    barely register on any of these.
    """
    lw = league_weight(stint["league"])
    if lw is None:
        return 0.0

    years_ago = max(0.0, as_of.year - float(stint.get("end", as_of.year)))
    recency = math.exp(-math.log(2) * years_ago / config.MANAGER_PRIOR_HALFLIFE_YEARS)

    matches = stint.get("matches")
    if matches is None:
        # Fall back to elapsed seasons if the scraper has not filled this in.
        seasons = max(0.0, float(stint.get("end", 0)) - float(stint.get("start", 0)))
        matches = seasons * 38
    # Effective experience: a match in a top European league counts for more
    # toward "proven" than the same match in a weak one.
    effective_matches = matches * lw
    sample = min(effective_matches / config.MANAGER_FULL_CONFIDENCE_MATCHES, 1.0)

    return float(lw * recency * sample)


def _rescale(value: float, league_average: float, weight: float) -> float:
    """
    Translate a per-match rate from one league into Premier League terms.

    Only the deviation from average carries over, discounted by league
    strength. Works in both directions: an outstanding defensive record in a
    weak league shrinks toward average here, exactly as an outstanding
    attacking one does.
    """
    return league_average + (value - league_average) * weight


def _seasonal_xg(stats: dict, as_of: pd.Timestamp) -> tuple[float, float]:
    """
    A stint's xG shape, recency-weighted across its own seasons.

    ppg is deliberately left as a flat aggregate elsewhere in this function --
    it answers "how successful were they overall" and a whole-stint sample
    size is the right input for that. xG is answering a different question,
    "what do they look like now", so a manager's most recent season at a club
    should count for more than their first. Same half-life as stint_weight()
    uses to compare stints against each other -- one philosophy for "how
    stale is this data", not two.
    """
    seasons = stats.get("xg_seasons")
    if not seasons:
        return stats.get("xg_for", config.LEAGUE_AVG_XG), stats.get("xg_against", config.LEAGUE_AVG_XG)

    num_for = num_against = denom = 0.0
    for s in seasons:
        years_ago = max(0.0, as_of.year - float(s["end"]))
        w = math.exp(-math.log(2) * years_ago / config.MANAGER_PRIOR_HALFLIFE_YEARS)
        num_for += w * s["xg_for"]
        num_against += w * s["xg_against"]
        denom += w
    if denom <= 0:
        return config.LEAGUE_AVG_XG, config.LEAGUE_AVG_XG
    return num_for / denom, num_against / denom


def manager_prior(team: str, stint_stats: dict[str, dict], as_of: pd.Timestamp,
                  prior_stints: list[dict] | None = None) -> dict | None:
    """
    Collapse a manager's previous jobs into one prior, scaled to EPL terms.

    stint_stats maps club name -> {"ppg", "matches"?, "xg_for"?/"xg_against"?
    (single season) or "xg_seasons"? (list of {"end", "xg_for", "xg_against"}
    for a multi-season stint, recency-weighted by _seasonal_xg) }, manually
    sourced from Transfermarkt manager-history pages (FBref/Understat are
    blocked). Only "ppg" is required; missing xG data defaults to league
    average (neutral -- no fabricated shape signal), missing "matches" falls
    back to an elapsed-season estimate. Returns None when nothing usable
    survives, which is the honest answer for an international-only CV.
    """
    # prior_stints defaults to the live 2026/27 config; the backtest passes a
    # historical list so this same function can price a past season's manager.
    if prior_stints is None:
        mgr = config.MANAGERS.get(team)
        prior_stints = mgr.get("prior_stints") if mgr else None
    if not prior_stints:
        return None

    num = {"ppg": 0.0, "xg_for": 0.0, "xg_against": 0.0}
    denom = 0.0
    used = []

    for stint in prior_stints:
        stats = stint_stats.get(stint["club"])
        if not stats:
            continue
        merged = {**stint, "matches": stats.get("matches", stint.get("matches"))}
        w = stint_weight(merged, as_of)
        if w <= 0:
            continue
        lw = league_weight(stint["league"]) or 0.0
        # Scale the margin over league average, never the raw figure. Points
        # per game is anchored at ~1.36 in every league, because the points on
        # offer are fixed -- so multiplying it by a league weight would drag a
        # title-winning record down to relegation form. What does not transfer
        # is the size of the edge: dominating a weak league is worth less.
        xg_for, xg_against = _seasonal_xg(stats, as_of)
        num["ppg"] += w * _rescale(stats["ppg"], config.LEAGUE_AVG_PPG, lw)
        num["xg_for"] += w * _rescale(xg_for, config.LEAGUE_AVG_XG, lw)
        num["xg_against"] += w * _rescale(xg_against, config.LEAGUE_AVG_XG, lw)
        denom += w
        used.append({"club": stint["club"], "league": stint["league"], "weight": round(w, 4)})

    if denom <= 0:
        return None

    return {
        "manager": (config.MANAGERS.get(team) or {}).get("name", team),
        "ppg": num["ppg"] / denom,
        "xg_for": num["xg_for"] / denom,
        "xg_against": num["xg_against"] / denom,
        "confidence": min(denom, 1.0),
        "stints_used": used,
    }


def blend_prior(prior_stat: float, current_stat: float, matches_at_club: int) -> float:
    """
    Hand over from prior to observed record as evidence accumulates.

    Below MANAGER_BLEND_MATCHES the manager's history still carries weight;
    above it, what they have actually done at this club takes over entirely.
    """
    if prior_stat is None or not np.isfinite(prior_stat):
        return current_stat
    alpha = min(matches_at_club / config.MANAGER_BLEND_MATCHES, 1.0)
    return (1.0 - alpha) * prior_stat + alpha * current_stat


def classify_style(xg_for: float, xg_against: float, league_xg_avg: float) -> str:
    """Descriptive label, for reporting rather than for the model."""
    hi, lo = 1.1 * league_xg_avg, 0.9 * league_xg_avg
    if xg_for > hi and xg_against < lo:
        return "Dominant"
    if xg_for > hi:
        return "Attacking"
    if xg_for <= lo and xg_against < lo:
        return "Defensive"
    return "Balanced"


# ------------------------------------------------------------ player adaptation


def adaptation_factor(
    epl_matches: int,
    source_league: str,
    minutes_last_season: float = 1800,
    recovery: bool | None = None,
    player_quality: float | None = None,
) -> float:
    """
    Discount applied to a player's market value until they are proven here.

    Transfermarkt prices talent, not Premier League output. A EUR 50m arrival
    from Ligue 1 and a EUR 50m player with 200 English appearances are not
    interchangeable in their first season, and the model should not treat
    them as such.

    player_quality (0..1, from debutant_career.quality_lookup) is a real
    individual signal -- how well THIS player actually performed pre-EPL, not
    just how strong their league is on average -- so two players priced the
    same from the same league no longer have to be treated identically. Only
    available for this season's named debutants; everyone else falls back to
    the league-only guess, as before.

    A good player_quality does NOT skip the league penalty: shining in a weak
    league still needs discounting the same as anyone else's move from that
    league would, so the two multiply rather than average. A stellar rating
    from a lw=0.9 league keeps most of its value; the same rating from a
    lw=0.15 league is discounted hard, exactly as it should be.
    """
    recovery = config.ADAPTATION_RECOVERY if recovery is None else recovery

    if epl_matches >= config.EPL_PROVEN_MATCHES:
        return 1.0

    lw = league_weight(source_league)
    if lw is None:
        # Unknown provenance: assume modest, do not assume Premier League ready.
        lw = 0.15
    base = float(lw)

    if player_quality is not None:
        base = player_quality * lw
    elif minutes_last_season < config.LOW_MINUTES_THRESHOLD:
        # Little senior football anywhere, so there is less to translate.
        # Skipped when a real rating is available -- that already reflects
        # genuine senior game time; this branch exists for the blind case.
        base *= config.LOW_MINUTES_PENALTY

    if epl_matches > 0:
        # Partial English experience closes part of the gap already.
        base += (1.0 - base) * (epl_matches / config.EPL_PROVEN_MATCHES)

    if recovery:
        # Adaptation improves through the year. With no mid-season refit, each
        # player contributes the season average of that curve.
        base = (base + 1.0) / 2.0

    return float(min(base, 1.0))


def squad_strength(players: pd.DataFrame, league_tier1: dict[str, float] | None = None) -> dict:
    """
    Adaptation-adjusted squad value.

    Weighted by value, not headcount: three unproven signings matter far more
    when they are the spine of the side than when they are squad filler.

    players needs: market_value_m, epl_matches, source_league,
    minutes_last_season. A "player" column, if present, is used to look up a
    real pre-EPL rating for this season's debutants (debutant_career).
    """
    if players.empty:
        return {"strength": float("nan"), "raw": float("nan"), "unproven_share": float("nan")}

    df = players.copy()
    quality = debutant_career.quality_lookup() if "player" in df.columns else {}
    leagues = debutant_career.league_lookup() if "player" in df.columns else {}
    df["adaptation"] = [
        adaptation_factor(
            int(r.epl_matches or 0),
            leagues.get(getattr(r, "player", None), r.source_league),
            float(r.minutes_last_season or 0),
            player_quality=quality.get(getattr(r, "player", None)),
        )
        for r in df.itertuples(index=False)
    ]
    df["adj_value"] = df["market_value_m"] * df["adaptation"]

    # The XI plus core rotation. Without a cut-off, academy names pad the sum.
    top = df.nlargest(config.SQUAD_DEPTH_COUNT, "adj_value")
    raw_top = df.nlargest(config.SQUAD_DEPTH_COUNT, "market_value_m")

    depth = squad_depth(df, league_tier1)

    unproven = top[top["epl_matches"].fillna(0) < config.EPL_PROVEN_MATCHES]

    # Never played here at all, as opposed to merely short of the proven
    # threshold. The distinction matters downstream: a club's fitted record
    # already contains everyone who played a Premier League minute for it, so
    # only a genuine newcomer is information the fit cannot have.
    #
    # A player whose name did not resolve to an FPL id has an unknown record,
    # not a zero one, and is excluded from both sides of the ratio rather than
    # counted as new.
    known = top if "fpl_matched" not in top.columns else top[top["fpl_matched"]]
    if known["market_value_m"].sum() > 0:
        newcomers = known[known["epl_matches"].fillna(0) == 0]
        # A newcomer with a real pre-EPL rating (debutant_career) is no
        # longer "information the fit cannot have" -- run_pipeline already
        # prices him into the club's squad_rating blend. Charging
        # ADAPTATION_DRAG for him too would be the same double-count this
        # project has caught before (Hull's 11 points, Sunderland's 19th):
        # one real signal, two penalties. Only a genuine unknown -- no
        # rating anywhere -- should still cost uncertainty here.
        rated = debutant_career.quality_lookup()
        if "player" in newcomers.columns:
            newcomers = newcomers[~newcomers["player"].isin(rated)]
        newcomer_share = float(newcomers["market_value_m"].sum() / known["market_value_m"].sum())
        coverage = float(known["market_value_m"].sum() / top["market_value_m"].sum())
    else:
        # Nothing measurable in this squad; say so instead of returning 0.0,
        # which a caller would read as "no newcomers".
        newcomer_share, coverage = float("nan"), 0.0

    return {
        "strength": float(top["adj_value"].sum()),
        "raw": float(raw_top["market_value_m"].sum()),
        "depth_ratio": depth["depth_ratio"],
        "depth_by_line": depth["by_line"],
        "weakest_line": depth["weakest_line"],
        "unproven_share": float(unproven["market_value_m"].sum() / raw_top["market_value_m"].sum()),
        "newcomer_share": newcomer_share,
        "known_coverage": coverage,
        "n_players": len(df),
    }


def squad_depth(df: pd.DataFrame, league_tier1: dict[str, float] | None = None) -> dict:
    """
    How far the quality falls from a club's first choice to its cover.

    Ranked by value WITHIN each line, not across the whole squad. A club can
    look deep on a flat top-18 count while having one senior goalkeeper and
    two centre-backs -- and it is precisely that shape, not the headline
    total, that decides what happens when a Thursday-night European tie lands
    between two league games.

    Depth per line is the SMALLER of two ratios, and it needs both:

      - the cover against the club's own first choice: how much is actually
        lost when the first pick sits out;
      - the cover against a league-average first choice for that line: whether
        the replacement is a Premier League starter at all.

    Taking only the first would call a squad of eleven interchangeable
    Championship players the deepest in the league -- their tier-2 nearly
    matches their tier-1 because nobody in it is any good. Taking only the
    second would just re-measure spending. A club is deep only when its cover
    is close to its own standard AND respectable by the league's.

    league_tier1 maps line -> mean tier-1 value across the league. Without it
    only the own-club ratio is applied, which is correct for a single squad in
    isolation but will over-rate a uniformly weak one.

    Returns depth_ratio on a 0-1 scale: 0 means no cover at all, 1.0 means the
    second string is as good as the first by both measures. Real squads sit
    well below 1.

    Needs a position column; without one this cannot be computed and says so
    rather than falling back to a position-blind number that would read as if
    it meant the same thing.
    """
    if "position" not in df.columns or df["position"].isna().all():
        return {"depth_ratio": float("nan"), "by_line": {}, "weakest_line": None, "tier1": {}}

    by_line: dict[str, float] = {}
    tier1_value: dict[str, float] = {}

    for line, starters in config.SQUAD_LINES.items():
        line_players = df[df["position"].isin(config.POSITION_LINES[line])]
        ranked = line_players.nlargest(starters * 2, "adj_value")["adj_value"].tolist()

        first = ranked[:starters]
        second = ranked[starters:starters * 2]
        # A line short of even a full first choice is padded with zeros: the
        # missing player is real absence, not a reason to skip the line.
        first += [0.0] * (starters - len(first))
        second += [0.0] * (starters - len(second))

        t1, t2 = sum(first), sum(second)
        own = (t2 / t1) if t1 > 0 else 0.0

        if league_tier1 and league_tier1.get(line, 0) > 0:
            vs_league = t2 / league_tier1[line]
            by_line[line] = float(min(own, vs_league))
        else:
            by_line[line] = float(own)
        tier1_value[line] = float(t1)

    total_t1 = sum(tier1_value.values())
    if total_t1 <= 0:
        return {"depth_ratio": float("nan"), "by_line": by_line, "weakest_line": None, "tier1": tier1_value}

    depth_ratio = sum(by_line[l] * tier1_value[l] for l in by_line) / total_t1
    weakest = min(by_line, key=by_line.get)
    return {
        "depth_ratio": float(depth_ratio),
        "by_line": by_line,
        "weakest_line": weakest,
        "tier1": tier1_value,
    }


def league_tier1_means(squads: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Mean first-choice value per line across the league.

    The yardstick squad_depth() needs to tell a deep squad apart from a
    uniformly weak one. Computed from the same adjusted values, so a league
    where everyone's newcomers are discounted shifts the bar consistently.
    """
    totals: dict[str, list[float]] = {line: [] for line in config.SQUAD_LINES}
    for df in squads.values():
        if "position" not in df.columns:
            continue
        for line, starters in config.SQUAD_LINES.items():
            line_players = df[df["position"].isin(config.POSITION_LINES[line])]
            first = line_players.nlargest(starters, "adj_value")["adj_value"].tolist()
            first += [0.0] * (starters - len(first))
            totals[line].append(sum(first))
    return {line: float(np.mean(v)) if v else 0.0 for line, v in totals.items()}


def squad_to_strength(squad_value: float, league_average: float) -> float:
    """
    Map squad value onto the attack/defence scale, for clubs with no usable
    Premier League record of their own.

    Log-linear: doubling squad value adds a fixed amount of strength rather
    than doubling it. The elasticity is calibrated in backtest.py.
    """
    if not np.isfinite(squad_value) or squad_value <= 0 or league_average <= 0:
        return 0.0
    return float(config.SQUAD_VALUE_ELASTICITY * math.log(squad_value / league_average))


# ------------------------------------------------------------ fatigue


@lru_cache(maxsize=1)
def _real_fixture_dates() -> dict[str, list[str]]:
    """EFL Cup + UCL/UEL/UECL dates actually scraped (parse_european_schedule)."""
    path = config.DATA / "european_fixtures_2026_27.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def european_match_dates(european: dict[str, str] | None = None) -> dict[str, list[pd.Timestamp]]:
    """
    Midweek European (and EFL Cup) dates per club, placed on the real calendar.

    Uses real scraped fixture dates (data/european_fixtures_2026_27.json, see
    src/parse_european_schedule.py) when available -- covers EFL Cup for
    every club still in it plus real UCL/UEL/UECL matchday dates for our 9
    entrants, not just the competition-level guess. Falls back to
    config.EUROPEAN_WEEKS (same date reused for every club in a competition,
    modelled as the Wednesday of each scheduled UEFA week) for a club with no
    real data, so the shape of the calendar still counts for fatigue even
    before/without a scrape.
    """
    european = config.EUROPEAN_COMPETITION if european is None else european
    real = _real_fixture_dates()
    out: dict[str, list[pd.Timestamp]] = {}

    for team in set(european) | set(real):
        if team in real:
            out[team] = [pd.Timestamp(d) for d in real[team]]
            continue
        weeks = config.EUROPEAN_WEEKS.get(european.get(team, ""))
        if weeks:
            out[team] = [pd.Timestamp(w) for w in weeks]
    return out


def build_fatigue(
    fixtures: pd.DataFrame,
    depth_ratios: dict[str, float] | None = None,
    european: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Per-fixture strength multipliers from schedule load and squad depth.

    A congested run costs a thin squad far more than a deep one: the same
    eleven keep playing, tire, and start breaking down. Injuries are not
    modelled directly -- this is the channel through which they arrive.

    Adds home_fatigue / away_fatigue columns.
    """
    if not config.FATIGUE_ENABLED:
        fixtures = fixtures.copy()
        fixtures["home_fatigue"] = 1.0
        fixtures["away_fatigue"] = 1.0
        return fixtures

    depth_ratios = depth_ratios or {}
    european = config.EUROPEAN_COMPETITION if european is None else european

    df = fixtures.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ignore_index=True)

    # Every team's league match dates, so rest and 28-day load can be measured.
    schedule: dict[str, list[pd.Timestamp]] = {}
    for row in df.itertuples(index=False):
        schedule.setdefault(row.home_team, []).append(row.date)
        schedule.setdefault(row.away_team, []).append(row.date)

    # European nights are placed on the calendar, not averaged into it.
    #
    # Spreading them as a flat load across 38 weeks made rest days read as if
    # the club had a free midweek -- a side playing Wednesday in Europe and
    # Saturday in the league showed a seven-day gap, so the rest penalty came
    # out at zero for exactly the clubs it exists to charge. Real European
    # weeks also cluster: the league phase runs September to January, then
    # knockouts, rather than trickling evenly through May.
    euro_dates = european_match_dates(european)
    for team, dates in euro_dates.items():
        if team in schedule:
            schedule[team].extend(dates)
            schedule[team].sort()

    def load(team: str, date: pd.Timestamp) -> tuple[float, float]:
        dates = schedule.get(team, [])
        prev = [d for d in dates if d < date]
        rest = (date - max(prev)).days if prev else 14
        window = sum(1 for d in dates if 0 < (date - d).days <= config.CONGESTION_WINDOW_DAYS)
        return rest, window

    # The league's own depth distribution sets the scale. A lone club (or a
    # test fixture) has no distribution to speak of, so fall back to a spread
    # wide enough that nobody is scored as an outlier against themselves.
    supplied = [v for v in depth_ratios.values() if np.isfinite(v)]
    league_mean_depth = float(np.mean(supplied)) if supplied else 0.5
    depth_spread = float(np.std(supplied)) if len(supplied) > 1 else 0.25
    if depth_spread < 1e-6:
        depth_spread = 0.25

    def penalty(team: str, date: pd.Timestamp) -> float:
        rest, congestion = load(team, date)

        rest_deficit = max(0.0, config.FATIGUE_REST_BASELINE - rest) / config.FATIGUE_REST_BASELINE
        # Four league games in 28 days is the normal rhythm; above that bites.
        congestion_load = max(0.0, congestion - 4.0) / 4.0

        # depth_ratio: per-line second choice over first choice, 0 to 1.
        # Judged against the league itself rather than a fixed threshold --
        # what matters for fatigue is being thinner than the clubs you play,
        # and the mean moves as squads are valued differently year to year.
        ratio = depth_ratios.get(team, league_mean_depth)
        thinness = float(np.clip(0.5 - (ratio - league_mean_depth) / (2 * depth_spread), 0.0, 1.0))

        raw = 0.5 * rest_deficit + 0.5 * min(congestion_load, 1.0)
        return float(config.FATIGUE_MAX_PENALTY * raw * (0.4 + 0.6 * thinness))

    df["home_fatigue"] = [1.0 - penalty(r.home_team, r.date) for r in df.itertuples(index=False)]
    df["away_fatigue"] = [1.0 - penalty(r.away_team, r.date) for r in df.itertuples(index=False)]
    return df


# ------------------------------------------------------------ self-check

if __name__ == "__main__":
    # League weights
    assert league_weight("Premier League") == 1.0
    assert league_weight("Bundesliga") == 0.7866
    assert league_weight("International") is None, "international must stay None"
    assert league_weight("Nonexistent League") is None

    # Adaptation: proven players are untouched, everyone else is discounted.
    assert adaptation_factor(200, "Bundesliga") == 1.0
    assert adaptation_factor(38, "La Liga") == 1.0
    bund = adaptation_factor(0, "Bundesliga")
    champ = adaptation_factor(0, "Championship")
    saudi = adaptation_factor(0, "Saudi Pro League")
    assert 0.85 < bund < 0.95, f"Bundesliga arrival: {bund:.3f}"
    assert champ < bund, "Championship must be discounted harder than Bundesliga"
    assert saudi < champ, "Saudi Pro League must be discounted harder still"
    # Partial Premier League experience should sit between the extremes.
    half = adaptation_factor(19, "Bundesliga")
    assert bund < half < 1.0, f"partial EPL experience not interpolating: {half:.3f}"
    # Barely any senior minutes is a further discount.
    assert adaptation_factor(0, "Bundesliga", minutes_last_season=200) < bund

    # Squad strength: value-weighted, not headcount-weighted.
    proven = pd.DataFrame(
        {
            "market_value_m": [60.0] * 18,
            "epl_matches": [100] * 18,
            "source_league": ["Premier League"] * 18,
            "minutes_last_season": [2500] * 18,
        }
    )
    # Same money, but the three most expensive players are new to England.
    spine_new = proven.copy()
    spine_new.loc[:2, ["epl_matches", "source_league"]] = [0, "Championship"]
    spine_new.loc[:2, "market_value_m"] = 90.0
    proven.loc[:2, "market_value_m"] = 90.0

    bench_new = proven.copy()
    bench_new.loc[15:, ["epl_matches", "source_league"]] = [0, "Championship"]

    s_proven = squad_strength(proven)["strength"]
    s_spine = squad_strength(spine_new)["strength"]
    s_bench = squad_strength(bench_new)["strength"]
    assert s_spine < s_bench < s_proven, (
        f"unproven core must cost more than unproven bench: "
        f"spine={s_spine:.0f} bench={s_bench:.0f} proven={s_proven:.0f}"
    )

    # A squad of players with one Premier League season behind them is not new
    # to the league, however far short of the proven threshold they sit.
    one_season = pd.DataFrame(
        {
            "market_value_m": [30.0] * 18,
            "epl_matches": [25] * 18,          # a full season, still under EPL_PROVEN_MATCHES
            "source_league": ["Premier League"] * 18,
            "minutes_last_season": [2000] * 18,
            "fpl_matched": [True] * 18,
        }
    )
    r_one = squad_strength(one_season)
    assert r_one["unproven_share"] == 1.0, "these players are indeed short of proven"
    assert r_one["newcomer_share"] == 0.0, "but none of them is new to the league"

    # An unresolved name is an unknown record, never counted as a newcomer.
    unknown = one_season.copy()
    unknown.loc[:5, "epl_matches"] = 0
    unknown.loc[:5, "fpl_matched"] = False
    r_unknown = squad_strength(unknown)
    assert r_unknown["newcomer_share"] == 0.0, "unmatched names must not become newcomers"
    assert r_unknown["known_coverage"] < 1.0, "coverage must report the unresolved share"

    # Manager prior: Jaissle, one UEFA stint plus one non-UEFA stint.
    stats = {
        "Red Bull Salzburg": {"ppg": 2.35, "xg_for": 2.10, "xg_against": 0.95, "matches": 76},
        "Al-Ahli": {"ppg": 2.20, "xg_for": 2.05, "xg_against": 1.00, "matches": 100},
    }
    prior = manager_prior("Newcastle Utd", stats, pd.Timestamp("2026-08-03"))
    assert prior is not None, "Jaissle prior should exist"
    # A title-winning record in a weaker league should land above average but
    # well short of what the raw number claims.
    assert config.LEAGUE_AVG_PPG < prior["ppg"] < 2.0, f"ppg not rescaled sanely: {prior['ppg']:.2f}"
    assert prior["xg_against"] < config.LEAGUE_AVG_XG, "good defence should stay below average"
    assert prior["xg_for"] > config.LEAGUE_AVG_XG, "good attack should stay above average"

    # Dominating a weak league must translate to less than the same record in
    # a strong one.
    weak = manager_prior(
        "Newcastle Utd",
        {"Red Bull Salzburg": {"ppg": 2.35, "xg_for": 2.1, "xg_against": 0.95, "matches": 76}},
        pd.Timestamp("2026-08-03"),
    )
    strong = manager_prior(
        "Chelsea",
        {"Bayer Leverkusen": {"ppg": 2.35, "xg_for": 2.1, "xg_against": 0.95, "matches": 76}},
        pd.Timestamp("2026-08-03"),
    )
    assert strong["ppg"] > weak["ppg"], (
        f"Bundesliga record should outweigh Austrian one: {strong['ppg']:.2f} vs {weak['ppg']:.2f}"
    )

    # ppg-only stint (no xG on Transfermarkt): must not crash, xG stays neutral.
    ppg_only = manager_prior(
        "Bournemouth",
        {"RB Leipzig": {"ppg": 1.90, "matches": 90}},
        pd.Timestamp("2026-08-03"),
    )
    assert ppg_only is not None
    assert abs(ppg_only["xg_for"] - config.LEAGUE_AVG_XG) < 1e-9, "missing xG must default neutral, not 0"

    # An international-only CV yields no usable prior, and says so.
    assert manager_prior("Brentford", {"Republic of Ireland": stats["Al-Ahli"]},
                         pd.Timestamp("2026-08-03")) is None

    # Within a stint, a declining season should count for more than a flat
    # mean would give it -- that's the whole point of weighting seasons.
    declining = {"Red Bull Salzburg": {
        "ppg": 2.0, "matches": 90,
        "xg_seasons": [
            {"end": 2022, "xg_for": 2.20, "xg_against": 0.80},   # strong, old
            {"end": 2026, "xg_for": 1.60, "xg_against": 1.50},   # weak, recent
        ],
    }}
    recent = manager_prior("Newcastle Utd", declining, pd.Timestamp("2026-08-03"))
    flat_mean_xga = (0.80 + 1.50) / 2
    assert recent["xg_against"] > config.LEAGUE_AVG_XG + (flat_mean_xga - config.LEAGUE_AVG_XG) * 0.5, (
        f"recent bad season should pull xGA up past a flat mean: {recent['xg_against']:.3f}"
    )
    # A single-season stint (flat xg_for/xg_against, no xg_seasons key) must
    # still resolve to exactly that value -- nothing to recency-weight.
    single = manager_prior("Newcastle Utd", {"Red Bull Salzburg": {
        "ppg": 2.0, "matches": 90, "xg_for": 1.9, "xg_against": 1.0,
    }}, pd.Timestamp("2026-08-03"))
    expected_for, _ = _seasonal_xg({"xg_for": 1.9, "xg_against": 1.0}, pd.Timestamp("2026-08-03"))
    assert expected_for == 1.9, "single-season stint must pass its own xG through unweighted"

    # Blending hands over to observed form as matches accumulate.
    assert blend_prior(1.0, 2.0, 0) == 1.0
    assert blend_prior(1.0, 2.0, 15) == 2.0
    assert blend_prior(1.0, 2.0, 30) == 2.0
    assert abs(blend_prior(1.0, 2.0, 5) - 1.3333) < 1e-3

    # Squad depth is per-line: a club stacked in one position is not deep.
    def _squad(rows):
        d = pd.DataFrame(rows, columns=["position", "market_value_m"])
        d["adj_value"] = d["market_value_m"]
        return d

    balanced = _squad(
        [("GK", 40), ("GK", 30)]
        + [("CB", 50), ("CB", 45), ("LB", 40), ("RB", 40), ("CB", 38), ("LB", 32), ("RB", 30), ("CB", 28)]
        + [("DM", 50), ("CM", 48), ("AM", 45), ("CM", 40), ("DM", 36), ("AM", 34)]
        + [("LW", 50), ("CF", 48), ("RW", 45), ("CF", 40), ("LW", 36), ("RW", 34)]
    )
    # Same money, same headcount -- but hoarded in defence, bare everywhere else.
    lopsided = _squad(
        [("GK", 40)]
        + [("CB", 50), ("CB", 48), ("LB", 45), ("RB", 44), ("CB", 42), ("LB", 40), ("RB", 38), ("CB", 36)]
        + [("DM", 50), ("CM", 48), ("AM", 45)]
        + [("LW", 50), ("CF", 48), ("RW", 45)]
    )
    d_bal = squad_depth(balanced)
    d_lop = squad_depth(lopsided)
    assert d_bal["depth_ratio"] > d_lop["depth_ratio"], (
        f"balanced cover must beat hoarding one line: {d_bal['depth_ratio']:.3f} vs {d_lop['depth_ratio']:.3f}"
    )
    # No cover anywhere outside defence, so those lines score zero and the
    # weakest line is one of them -- never the stacked one.
    assert d_lop["by_line"]["DEF"] > 0, "defence does have cover here"
    assert d_lop["by_line"]["ATT"] == 0.0, "attack has no second choice at all"
    assert d_lop["weakest_line"] in ("GK", "MID", "ATT"), d_lop["weakest_line"]
    assert 0.0 <= d_bal["depth_ratio"] <= 1.0, "depth ratio must stay on its 0-1 scale"

    # A uniformly cheap squad has a superb own-club ratio -- every player is
    # equally poor -- but must not out-rank a strong squad once the league
    # standard is applied. This is the flaw the vs-league term exists to fix.
    flat_cheap = _squad(
        [("GK", 3), ("GK", 3)]
        + [("CB", 4), ("CB", 4), ("LB", 4), ("RB", 4), ("CB", 3), ("LB", 3), ("RB", 3), ("CB", 3)]
        + [("DM", 4), ("CM", 4), ("AM", 4), ("CM", 3), ("DM", 3), ("AM", 3)]
        + [("LW", 4), ("CF", 4), ("RW", 4), ("CF", 3), ("LW", 3), ("RW", 3)]
    )
    league_bar = league_tier1_means({"a": balanced, "b": flat_cheap})
    assert squad_depth(flat_cheap)["depth_ratio"] > 0.7, "own-club ratio alone flatters a weak squad"
    assert squad_depth(flat_cheap, league_bar)["depth_ratio"] < squad_depth(balanced, league_bar)["depth_ratio"], (
        "league standard must stop a uniformly cheap squad ranking as deep"
    )

    # Without a position column the metric is not computable, and says so
    # rather than quietly returning a position-blind number.
    no_pos = pd.DataFrame({"market_value_m": [50.0] * 18, "adj_value": [50.0] * 18})
    assert np.isnan(squad_depth(no_pos)["depth_ratio"]), "missing positions must yield NaN, not a fallback"

    # Fatigue: thin squads suffer more from the same congested schedule.
    dates = pd.date_range("2026-12-01", periods=8, freq="3D")
    fx = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "home_team": ["Thin"] * 8 + ["Deep"] * 8,
            "away_team": ["X"] * 8 + ["Y"] * 8,
            "matchweek": list(range(8)) * 2,
        }
    )
    fat = build_fatigue(fx, depth_ratios={"Thin": 0.25, "Deep": 0.75})
    thin_pen = 1 - fat[fat.home_team == "Thin"]["home_fatigue"].mean()
    deep_pen = 1 - fat[fat.home_team == "Deep"]["home_fatigue"].mean()
    assert thin_pen > deep_pen > 0, f"depth not protecting: thin={thin_pen:.4f} deep={deep_pen:.4f}"
    assert thin_pen <= config.FATIGUE_MAX_PENALTY, "penalty exceeded its cap"

    print(
        f"OK  adaptation Bundesliga={bund:.3f} Championship={champ:.3f} Saudi={saudi:.3f} | "
        f"squad proven={s_proven:.0f} bench-new={s_bench:.0f} spine-new={s_spine:.0f} | "
        f"Jaissle prior ppg={prior['ppg']:.2f} conf={prior['confidence']:.2f} | "
        f"fatigue thin={thin_pen:.3f} deep={deep_pen:.3f}"
    )
