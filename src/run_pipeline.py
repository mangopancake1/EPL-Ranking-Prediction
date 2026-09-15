"""
End-to-end 2026/27 prediction.

    python -m src.run_pipeline

Four stages:

1. Fit Dixon-Coles on three seasons of real Premier League results.
2. Assemble a strength for all 20 clubs of 2026/27. Seventeen have a record to
   fit; the three promoted clubs have none and are priced off squad value.
3. Adjust for what the fit cannot know: a new manager in the dugout, a squad
   rebuilt with players who have never played here, and the fixture schedule.
4. Simulate the season 10,000 times and report the rank distribution.

Every adjustment is written to output/strengths_2026_27.csv with its own
column, so any number in the final table can be traced back to what moved it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src import championship, debutant_career, features, player_ratings
from src.dixon_coles import DCParams, fit_blended, time_weights
from src.monte_carlo import default_flags, rank_matrix, simulate, summarise


# ------------------------------------------------------------------ loading


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(config.MATCHES_FILE, parse_dates=["date"])
    df = df[df["season"].isin(config.TRAIN_SEASONS)]
    if df.empty:
        raise ValueError(f"no matches for seasons {config.TRAIN_SEASONS}")
    return df


def load_fixtures() -> pd.DataFrame:
    df = pd.read_csv(config.FIXTURES_FILE, parse_dates=["date"])
    unknown = (set(df["home_team"]) | set(df["away_team"])) - set(config.TEAMS)
    if unknown:
        raise ValueError(f"fixtures reference teams outside TEAMS: {unknown}")
    return df


def load_squads() -> pd.DataFrame:
    data = json.loads(config.SQUAD_VALUES_FILE.read_text(encoding="utf-8"))
    return pd.DataFrame(data["players"])


def load_priors() -> dict:
    if not config.MANAGER_PRIORS_FILE.exists():
        print("  ! no manager_priors.json -- running without the manager feature")
        return {}
    return json.loads(config.MANAGER_PRIORS_FILE.read_text(encoding="utf-8"))["priors"]


# ------------------------------------------------------------------ conversions


def ppg_to_strength_slope(matches: pd.DataFrame, fitted: DCParams, as_of: pd.Timestamp) -> float:
    """
    How much fitted strength is one point per game worth?

    Measured, not assumed: regress each fitted club's combined attack+defence
    on the points per game it actually took, weighted the same way the fit
    weights matches. A manager prior arrives as a points-per-game figure and
    has to be converted into the model's units somehow -- doing it from this
    data beats picking a constant that looks about right.
    """
    w = time_weights(matches["date"], as_of, config.XI)
    pts: dict[str, float] = {}
    games: dict[str, float] = {}

    for row, weight in zip(matches.itertuples(index=False), w):
        hg, ag = row.home_goals, row.away_goals
        h_pts = config.POINTS_WIN if hg > ag else (config.POINTS_DRAW if hg == ag else config.POINTS_LOSS)
        a_pts = config.POINTS_WIN if ag > hg else (config.POINTS_DRAW if hg == ag else config.POINTS_LOSS)
        for team, p in ((row.home_team, h_pts), (row.away_team, a_pts)):
            pts[team] = pts.get(team, 0.0) + weight * p
            games[team] = games.get(team, 0.0) + weight

    # A club with almost no weight left in the window carries no information
    # about the slope and would only add noise.
    rows = [
        (pts[t] / games[t], fitted.attack[i] + fitted.defence[i])
        for i, t in enumerate(fitted.teams)
        if games.get(t, 0.0) >= 5.0
    ]
    if len(rows) < 8:
        raise ValueError(f"only {len(rows)} clubs with usable weight -- cannot measure the slope")

    ppg = np.array([r[0] for r in rows])
    strength = np.array([r[1] for r in rows])
    slope, _ = np.polyfit(ppg, strength, 1)

    if not np.isfinite(slope) or slope <= 0:
        raise ValueError(f"implausible ppg->strength slope: {slope}")
    return float(slope)


def evidence_share(matches: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    """
    How much time-weighted match evidence the fit has per club, relative to a
    club that never left the division.

    Leeds and Sunderland came up for 2025-26 and have a single Premier League
    season on file; Arsenal has three. Both currently get the same narrow
    band, which says the model is equally sure about a one-season sample and a
    three-season one. It is not.
    """
    weights = time_weights(matches["date"], as_of, config.XI)
    per_club: dict[str, float] = {}
    for row, w in zip(matches.itertuples(index=False), weights):
        per_club[row.home_team] = per_club.get(row.home_team, 0.0) + w
        per_club[row.away_team] = per_club.get(row.away_team, 0.0) + w

    if not per_club:
        return {}
    ever_present = max(per_club.values())
    return {t: min(v / ever_present, 1.0) for t, v in per_club.items()}


def manager_tenure_share(team: str, matches: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """
    What fraction of a club's time-weighted training evidence was actually
    played under its current manager.

    The changeover shrink handles a brand-new appointment. It misses the
    quieter case: a manager who arrived partway through the window and is now
    "established", so their club's fitted strength still carries a predecessor
    the model will never correct for. Everton's window is more than half David
    Moyes's predecessor; Brighton's includes a full De Zerbi season.

    Returns 1.0 when the whole window belongs to the current manager (or when
    there is nothing to judge), so clubs like Arsenal are untouched.
    """
    since = config.MANAGERS.get(team, {}).get("since")
    if not since:
        return 1.0

    played = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    if played.empty:
        return 1.0

    weights = time_weights(played["date"], as_of, config.XI)
    total = float(weights.sum())
    if total <= 0:
        return 1.0

    under_current = float(weights[(played["date"] >= pd.Timestamp(since)).to_numpy()].sum())
    return under_current / total


def promoted_baseline(matches: pd.DataFrame, slope: float) -> tuple[float, int, float]:
    """
    Combined strength of a typical promoted club, measured from the real ones.

    Squad value cannot price a promoted club here. The value data covers each
    promoted club's whole squad but only the globally-expensive players at
    established clubs, so a cross-club ratio compares a full squad against a
    partial one and buries the promoted side.

    What the match data does contain is every club that came up during the
    training window -- present in one season, absent from the one before. What
    they went on to score is the honest answer, and it already includes the
    step up in level, so no separate promotion penalty is needed on top.
    """
    seasons = config.TRAIN_SEASONS
    present = {s: set(matches.loc[matches["season"] == s, "home_team"]) for s in seasons}

    ppgs = []
    for prev, cur in zip(seasons, seasons[1:]):
        for team in present[cur] - present[prev]:
            d = matches[(matches["season"] == cur) &
                        ((matches["home_team"] == team) | (matches["away_team"] == team))]
            pts = 0
            for r in d.itertuples(index=False):
                home = r.home_team == team
                gf, ga = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
                pts += config.POINTS_WIN if gf > ga else (config.POINTS_DRAW if gf == ga else config.POINTS_LOSS)
            ppgs.append(pts / len(d))

    if len(ppgs) < 3:
        raise ValueError(f"only {len(ppgs)} promoted club-seasons in the window -- not enough to measure")

    mean_ppg = float(np.mean(ppgs))
    combined = (mean_ppg - config.LEAGUE_AVG_PPG) * slope * config.PROMOTION_PENALTY
    return combined, len(ppgs), mean_ppg


def manager_delta(
    team: str, priors: dict, slope: float, as_of: pd.Timestamp,
    prior_stints: list[dict] | None = None,
) -> tuple[float, float, dict | None]:
    """
    Strength shift from the manager prior: real statistics first (ppg AND
    xG), which league they were earned in, then split between attack and
    defence by shape.

    ppg answers "how successful were they" -- results, including luck and
    finishing. xG answers "how good was the underlying performance" --
    harder to fake over a real sample. Both are genuine manager statistics
    and both now size the budget, not just ppg with xG merely reshaping it
    afterward. Blended in PPG UNITS, using a ratio measured from every
    stored prior stint's own spread (std of real ppg margins over std of
    real xG margins across the dataset) rather than a chosen constant --
    config.MANAGER_XG_TO_PPG.

    League quality (features.league_weight) already discounted both margins
    before they got here, and stint_weight() already discounted how much of
    a manager's experience even counts, twice over, by league quality and
    by recency. This function only combines what's left and splits it.

    Applied only where the base strength cannot already know the manager: a
    new appointment, or a promoted club priced purely off squad value.
    Re-applying it to a settled manager would count the same effect twice --
    their record at this club IS the fit.
    """
    stints = priors.get(team, {})
    prior = features.manager_prior(team, stints, as_of, prior_stints=prior_stints)
    if prior is None:
        return 0.0, 0.0, None

    ppg_margin = prior["ppg"] - config.LEAGUE_AVG_PPG
    has_xg = any(("xg_for" in s) or ("xg_seasons" in s) for s in stints.values())

    if has_xg:
        atk_margin = prior["xg_for"] - config.LEAGUE_AVG_XG
        def_margin = config.LEAGUE_AVG_XG - prior["xg_against"]  # positive = good defence
        xg_margin_ppg_units = (atk_margin + def_margin) * config.MANAGER_XG_TO_PPG
        margin = (ppg_margin + xg_margin_ppg_units) / 2.0
    else:
        margin = ppg_margin

    raw = config.MANAGER_EFFECT_SHARE * prior["confidence"] * margin * slope
    raw = float(np.clip(raw, -config.MANAGER_EFFECT_CAP, config.MANAGER_EFFECT_CAP))

    if not has_xg:
        return raw / 2.0, raw / 2.0, prior

    total_margin = atk_margin + def_margin

    # Only trust the split when attack and defence tell the same story (both
    # good or both weak). A manager who is sharp going forward but leaky at
    # the back has no honest way to redistribute the budget by shape -- the
    # two signals disagree, so fall back to even.
    if atk_margin * def_margin > 0 and abs(total_margin) > 0.05:
        atk_share = float(np.clip(atk_margin / total_margin, 0.15, 0.85))
    else:
        atk_share = 0.5

    atk_delta = raw * atk_share
    return atk_delta, raw - atk_delta, prior


# ------------------------------------------------------------------ assembly


def _club_ppg(played: pd.DataFrame, team: str) -> tuple[float, int]:
    """Points per game this club has actually managed so far, and games played."""
    d = played[(played["home_team"] == team) | (played["away_team"] == team)]
    if d.empty:
        return 0.0, 0
    pts = 0
    for r in d.itertuples(index=False):
        home = r.home_team == team
        gf, ga = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
        pts += config.POINTS_WIN if gf > ga else (config.POINTS_DRAW if gf == ga else config.POINTS_LOSS)
    return pts / len(d), len(d)


def build_strengths(
    matches: pd.DataFrame,
    squads: pd.DataFrame,
    priors: dict,
    as_of: pd.Timestamp,
    played: pd.DataFrame | None = None,
) -> tuple[DCParams, pd.DataFrame]:
    """
    Assemble one attack/defence pair per 2026/27 club, with an audit trail.

    played, if given, is this season's real results. The fitted clubs pick them
    up automatically -- they are already in `matches` by the time this is
    called, recency-weighted by the same decay as everything else. A promoted
    club gets no fitted rating worth using off a handful of games, so its
    baseline level is blended toward what those games actually said, the
    baseline's weight falling as games accumulate (config.LIVE_PRIOR_MATCH_WEIGHT).
    """
    print("fitting Dixon-Coles ...")
    fitted = fit_blended(matches, reference_date=as_of)
    print(f"  {fitted.n_matches} matches, {len(fitted.teams)} clubs, "
          f"home_adv={fitted.home_adv:.3f} rho={fitted.rho:.3f}")

    # home_adv is one fitted constant, but Premier League home advantage is on
    # a long slide (mid-2020s ~43% home wins, down from ~65% decades ago). The
    # time decay already tilts the fit toward recent seasons; this just makes
    # the trend visible, so a sharp ongoing drop is seen rather than averaged
    # away -- same spirit as the drag and calendar-horizon warnings below.
    trained = matches[matches["season"].isin(config.TRAIN_SEASONS)]
    hw = trained.groupby("season").apply(
        lambda d: (d["home_goals"] > d["away_goals"]).mean(), include_groups=False)
    print("  home-win rate by training season: "
          + ", ".join(f"{s.split('-')[0]} {r:.0%}" for s, r in hw.items()))
    if len(hw) >= 2 and hw.iloc[-1] < hw.iloc[0] - 0.03:
        print(f"  ! home advantage is trending down ({hw.iloc[0]:.0%} -> {hw.iloc[-1]:.0%} "
              "across the window); a single fitted home_adv slightly overstates it")

    slope = ppg_to_strength_slope(matches, fitted, as_of)
    print(f"  1.00 ppg = {slope:.3f} combined strength (measured from the fit)")

    # Squad aggregates for every club, promoted or not. Two passes: depth is
    # judged partly against the league's own first-choice standard, so that
    # standard has to exist before any club can be scored against it.
    quality = debutant_career.quality_lookup()
    leagues = debutant_career.league_lookup()
    by_club = {}
    for team in config.TEAMS:
        sub = squads[squads["club"] == team].copy()
        if sub.empty:
            raise ValueError(f"no squad-value rows for {team}")
        sub["adaptation"] = [
            features.adaptation_factor(int(r.epl_matches or 0),
                                       leagues.get(getattr(r, "player", None), r.source_league),
                                       float(r.minutes_last_season or 0),
                                       player_quality=quality.get(getattr(r, "player", None)))
            for r in sub.itertuples(index=False)
        ]
        sub["adj_value"] = sub["market_value_m"] * sub["adaptation"]
        by_club[team] = sub

    league_tier1 = features.league_tier1_means(by_club)
    print("  league first-choice value per line: "
          + ", ".join(f"{k} {v:.0f}m" for k, v in league_tier1.items()))

    squad = {team: features.squad_strength(by_club[team], league_tier1) for team in config.TEAMS}

    shares = np.array([squad[t]["newcomer_share"] for t in config.TEAMS], dtype=float)
    if np.isnan(shares).any():
        blind = [t for t, v in zip(config.TEAMS, shares) if np.isnan(v)]
        raise ValueError(f"no measurable EPL history for any listed player at: {blind}")
    league_avg_newcomer = float(shares.mean())

    # Promoted clubs are priced against each other, never against the
    # established sides: all three squads come from the same club-page source
    # and are complete, so the ratio between them is a fair comparison. The
    # level they sit at comes from the measured baseline below.
    promo_values = np.array([squad[t]["strength"] for t in config.PROMOTED])
    promo_avg_value = float(np.exp(np.log(promo_values).mean()))

    promo_base, n_promo, promo_ppg = promoted_baseline(matches, slope)
    print(f"  promoted baseline: {promo_ppg:.2f} ppg from {n_promo} real promoted club-seasons "
          f"-> {promo_base:+.3f} combined strength")

    # A promoted club's own Championship record, on top of the population
    # baseline. The line is measured from past promoted clubs (their second-tier
    # ppg in the promotion season against their PL ppg the season after); a
    # club with no record on file keeps the baseline. See src/championship.py.
    if config.CHAMPIONSHIP_OFFSET:
        champ_model, champ_ppg, champ_label = championship.choose(
            matches, championship.current_clubs(), config.TRAIN_SEASONS)
    else:
        champ_model, champ_ppg, champ_label = None, {}, "off"
    if champ_model is None:
        print(f"  championship offset: {'switched off in config' if not config.CHAMPIONSHIP_OFFSET else 'no usable fit'} -- baseline only")
    else:
        print(f"  championship offset [{champ_label}]: pl_ppg = {champ_model['a']:.2f} + "
              f"{champ_model['b']:.2f} * champ_{champ_model.get('input', 'ppg')}  "
              f"(r={champ_model['r']:.2f})"
              + ("" if champ_model["b"] == champ_model["b_raw"]
                 else f"  [raw slope {champ_model['b_raw']:.2f}, clamped]"))
        # Both large-sample inputs, so the better one is visible even when the
        # current clubs only have the weaker input on file.
        for inp in ("gd", "ppg"):
            alt = championship.fit_big(inp)
            if alt and alt.get("input") != champ_model.get("input"):
                print(f"    (PL-era {inp}: slope {alt['b']:.2f}, r={alt['r']:.2f}, n={alt['n']} -- "
                      + ("not applied, current clubs lack gd on file" if inp == "gd" else "for comparison") + ")")
        for t in config.PROMOTED:
            if t not in champ_ppg:
                print(f"    {t}: no Championship record on file, baseline only")
    worst_cov = min(squad[t]["known_coverage"] for t in config.TEAMS)
    print(f"  mean newcomer share {league_avg_newcomer:.2f} "
          f"(lowest club coverage {worst_cov:.0%} of listed value with a resolved record)")

    # Second strength estimate, from the current squad's real player ratings.
    # The fit only knows the squads that actually played those matches, so a
    # club that has since signed proven quality reads as unchanged. Blended
    # (not added) below -- see player_ratings' module docstring.
    as_of_year = as_of.year
    squad_ratings, rating_cover = {}, {}
    for team in config.TEAMS:
        q, n = player_ratings.squad_rating(by_club[team], as_of_year, club=team)
        rating_cover[team] = n
        if q is not None:
            squad_ratings[team] = q

    fitted_combined = {
        t: float(fitted.attack[fitted.index[t]] + fitted.defence[fitted.index[t]])
        for t in squad_ratings
        if t not in config.PROMOTED  # no fitted strength exists to regress against
    }
    rating_implied, rating_fit = player_ratings.implied_strength(squad_ratings, fitted_combined)
    if rating_implied:
        print(f"  squad rating vs fitted strength: r={rating_fit['r']:.2f} "
              f"(R2={rating_fit['r2']:.2f}, n={rating_fit['n']}), "
              f"1.0 rating = {rating_fit['slope']:.2f} combined strength "
              f"-> blended at {1 - config.SQUAD_RATING_BLEND:.0%}")
        # Promoted clubs are read off a line fitted without them. Say so when
        # one sits outside the range that line was fitted over, and by how
        # much -- an extrapolated club is a weaker claim than an interpolated one.
        for team in sorted(config.PROMOTED):
            q = squad_ratings.get(team)
            if q is None:
                print(f"  {team}: no squad rating, baseline only")
            elif not rating_fit["fit_lo"] <= q <= rating_fit["fit_hi"]:
                edge = rating_fit["fit_lo"] if q < rating_fit["fit_lo"] else rating_fit["fit_hi"]
                print(f"  ! {team} rating {q:.3f} is outside the fitted range "
                      f"{rating_fit['fit_lo']:.3f}-{rating_fit['fit_hi']:.3f} "
                      f"by {abs(q - edge):.3f} -- extrapolated")

    rows = []
    for team in config.TEAMS:
        s = squad[team]
        promoted = team in config.PROMOTED
        mgr = config.MANAGERS.get(team, {})

        if promoted:
            # Level from what promoted clubs actually managed; separation
            # between the three from their squad values relative to each other.
            spread = features.squad_to_strength(s["strength"], promo_avg_value)
            champ_off = (championship.club_offset_ppg(team, champ_model, champ_ppg)
                         * slope * config.PROMOTION_PENALTY)
            combined = promo_base + champ_off + spread
            attack = defence = combined / 2.0
            source = "promoted"
            promo = promo_base

            # Weekly series only: condition the level on this club's real games.
            # A promoted club has no fitted rating worth trusting off 3 games,
            # so blend the population baseline with what those games said,
            # shrinking the baseline as evidence builds. This is the peer's
            # "prior in units of matches" -- LIVE_PRIOR_MATCH_WEIGHT is exactly
            # how many games it takes for the two to weigh equally.
            if played is not None and len(played):
                club_ppg, n_played = _club_ppg(played, team)
                if n_played > 0:
                    minifit = (club_ppg - config.LEAGUE_AVG_PPG) * slope * config.PROMOTION_PENALTY
                    w = config.LIVE_PRIOR_MATCH_WEIGHT / (config.LIVE_PRIOR_MATCH_WEIGHT + n_played)
                    combined = w * combined + (1.0 - w) * minifit
                    attack = defence = combined / 2.0

            # Blend in the squad-rating estimate, exactly as the fitted clubs
            # do. This was skipped while the rating file covered almost none
            # of a promoted squad; the debutant-career fallback changed that
            # (Hull 15 of 18 rated, Coventry 14, Ipswich 15). It matters most
            # here: a promoted club has no Premier League match evidence at
            # all, so the baseline is a league-wide average with a squad-value
            # spread, and the ratings are the only club-specific signal there is.
            rating_shift = 0.0
            if team in rating_implied:
                target = (config.SQUAD_RATING_BLEND * combined
                          + (1.0 - config.SQUAD_RATING_BLEND) * rating_implied[team])
                rating_shift = (target - combined) / 2.0
                combined = target
                attack = defence = combined / 2.0
            drag = 0.0
            shrink = 0.0
            foreign_share = 1.0  # the baseline knows nothing about who is in charge
        else:
            i = fitted.index[team]
            attack, defence = float(fitted.attack[i]), float(fitted.defence[i])
            promo = 0.0
            champ_off = 0.0
            source = "fitted"

            # Blend in the squad-rating estimate of the same strength. Only
            # the LEVEL moves; the club's own attack/defence shape is kept,
            # because the squad rating says how good this squad is, not
            # whether it is good going forward or at the back.
            rating_shift = 0.0
            if team in rating_implied:
                own = attack + defence
                target = (config.SQUAD_RATING_BLEND * own
                          + (1.0 - config.SQUAD_RATING_BLEND) * rating_implied[team])
                rating_shift = (target - own) / 2.0
                attack += rating_shift
                defence += rating_shift
            # The fit contains everyone who has played a Premier League minute
            # for this club. What it cannot contain is a player who has never
            # played in the league at all. Charged relative to the league mean,
            # because every club turns over some of its squad each summer and
            # the fitted baseline already carries a typical amount of it.
            drag = config.ADAPTATION_DRAG * (s["newcomer_share"] - league_avg_newcomer)
            attack -= drag
            defence -= drag

            # The fitted strength reflects whoever actually managed this club
            # through the training window -- not necessarily the incoming one.
            # Shrink toward league average before adding the new manager's own
            # prior, so a departed manager's contribution isn't kept in full
            # AND effectively double-counted underneath the replacement's.
            #
            # Scaled by how much of the window the current manager was NOT in
            # charge for, rather than by job title. A brand-new appointment
            # gets the full shrink because none of the window is theirs; a
            # manager who arrived halfway through gets half of it. This is what
            # catches Everton (more than half the window is Dyche's) and
            # Brighton (a full De Zerbi season), both of which the old
            # status-based test waved through as "established".
            foreign_share = 1.0 - manager_tenure_share(team, matches, as_of)
            shrink = config.MANAGER_CHANGEOVER_SHRINK * foreign_share
            if shrink > 1e-9:
                defence_mean = float(fitted.defence.mean())
                attack *= (1.0 - shrink)  # attack's own mean is 0 by the fit's identifiability constraint
                defence = defence_mean + (defence - defence_mean) * (1.0 - shrink)

        # Manager prior, weighted by how much of the club's record the current
        # manager did NOT produce.
        #
        # The gate used to be the job title -- only "new" or "interim_to_full"
        # got a prior at all -- which left an asymmetry: the shrink above
        # already removes the departed manager's share of the fit for every
        # club, but for an "established" manager nothing filled the gap it
        # left. Everton's window is more than half Dyche's and David Moyes has
        # 261 matches of prior record; none of it counted.
        #
        # Now any manager with a real prior contributes, scaled by
        # foreign_share: a brand-new appointment gets the full prior because
        # none of the window is theirs, a manager two years into the job gets
        # almost none because the fit already IS their record. Promoted clubs
        # take the full prior -- their baseline is measured from other clubs
        # entirely and knows nothing about who is in charge.
        atk_delta, def_delta, prior = manager_delta(team, priors, slope, as_of)
        weight = 1.0 if promoted else foreign_share
        atk_delta, def_delta = atk_delta * weight, def_delta * weight
        attack += atk_delta
        defence += def_delta

        rows.append(
            {
                "team": team,
                "source": source,
                "manager": mgr.get("name", ""),
                "manager_status": mgr.get("status", ""),
                "squad_value_m": s["raw"],
                "squad_adjusted_m": s["strength"],
                "unproven_share": s["unproven_share"],
                "newcomer_share": s["newcomer_share"],
                "known_coverage": s["known_coverage"],
                "depth_ratio": s["depth_ratio"],
                "n_players_listed": s["n_players"],
                "squad_rating": squad_ratings.get(team, np.nan),
                # The Dixon-Coles strength before any adjustment. Written out
                # because the dashboard scatter needs it: plotting the FINAL
                # strength against squad rating is circular -- the rating shift
                # has already pulled it toward the line the chart then draws.
                "fitted_combined": fitted_combined.get(team, np.nan),
                "rating_shift": rating_shift,
                "prior_ppg": prior["ppg"] if prior else np.nan,
                "prior_confidence": prior["confidence"] if prior else np.nan,
                "adaptation_drag": drag,
                "changeover_shrink": shrink,
                "promoted_baseline": promo,
                "championship_offset": champ_off,
                "manager_delta_atk": atk_delta,
                "manager_delta_def": def_delta,
                "manager_delta": atk_delta + def_delta,
                "attack": attack,
                "defence": defence,
            }
        )

    audit = pd.DataFrame(rows)

    # A per-club adjustment with no spread across clubs is not an adjustment.
    # This is the guard that caught ADAPTATION_DRAG going degenerate. It stays
    # on manager_delta and anything else that is meant to differentiate; drag
    # itself is now deliberately 0.0, so an all-flat reading there is expected,
    # not a warning.
    fitted_rows = audit[audit["source"] == "fitted"]
    checks = [("manager_delta", "manager delta")]
    if config.ADAPTATION_DRAG != 0.0:
        checks.append(("adaptation_drag", "adaptation drag"))
    for column, label in checks:
        if len(fitted_rows) > 1 and fitted_rows[column].std() < 1e-9:
            print(f"  ! {label} is flat across all fitted clubs "
                  f"({fitted_rows[column].iloc[0]:+.4f}) -- it separates nobody")

    # Re-centre attack over the 20 clubs actually playing. Shifting attack and
    # defence by the same constant leaves every lambda untouched -- it is a
    # pure gauge change -- so this only restores the fit's own convention after
    # relegated clubs dropped out and promoted ones came in.
    shift = audit["attack"].mean()
    audit["attack"] -= shift
    audit["defence"] -= shift

    params = DCParams(
        teams=list(audit["team"]),
        attack=audit["attack"].to_numpy(),
        defence=audit["defence"].to_numpy(),
        home_adv=fitted.home_adv,
        rho=fitted.rho,
        n_matches=fitted.n_matches,
        source=f"{fitted.source}+squad+manager",
    )
    return params, audit


# ------------------------------------------------------------------ run


def main(played: pd.DataFrame | None = None, write: bool = True
         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    played: this season's real results (src/live_results). None is the
    pre-season forecast. A DataFrame folds those results into the fit and
    simulates only the fixtures not yet played -- the weekly series.

    write: whether to overwrite output/*.csv. The weekly run passes False so it
    does not clobber the pre-season files; src/weekly.py logs it separately.
    """
    as_of = pd.Timestamp(config.TODAY)
    mode = "weekly-updating" if played is not None and len(played) else "pre-season"
    print(f"EPL {config.SEASON} prediction ({mode}), as of {config.TODAY}\n")

    matches = load_matches()
    fixtures = load_fixtures()
    squads = load_squads()
    priors = load_priors()

    if played is not None and len(played):
        # Fold the real results into the training data. fit_blended weights
        # them by the same time decay as every other match, so three weeks of
        # 2026/27 sit near full weight against three seasons that have faded.
        matches = pd.concat([matches, played[matches.columns]], ignore_index=True)
        done = set(zip(played["home_team"], played["away_team"]))
        fixtures = fixtures[~fixtures.apply(
            lambda r: (r.home_team, r.away_team) in done, axis=1)].reset_index(drop=True)
        print(f"conditioned on {len(played)} played matches "
              f"(through matchweek {int(played['matchweek'].max())}); "
              f"{len(fixtures)} fixtures remain")

    print(f"loaded {len(matches)} matches, {len(fixtures)} fixtures, "
          f"{len(squads)} players, {len(priors)} manager priors\n")

    params, audit = build_strengths(matches, squads, priors, as_of, played=played)

    depth = dict(zip(audit["team"], audit["depth_ratio"]))
    fixtures = features.build_fatigue(fixtures, depth_ratios=depth)
    worst = 1 - min(fixtures["home_fatigue"].min(), fixtures["away_fatigue"].min())
    print(f"\nfatigue applied, heaviest single-match penalty {worst:.1%}")

    # The midweek calendar is only as long as the draw that exists. UEFA
    # knockout rounds and the later EFL Cup rounds are not scheduled yet, so
    # the fatigue model sees an empty spring and charges nobody for it. That is
    # a data horizon, not an easier run-in, and it should be visible every run.
    midweek = features.european_match_dates()
    known = [d for dates in midweek.values() for d in dates]
    if known:
        horizon = max(known)
        beyond = int((fixtures["date"] > horizon).sum())
        if beyond:
            print(f"  ! midweek fixtures known only to {horizon:%d %b %Y}; "
                  f"{beyond} of {len(fixtures)} league matches "
                  f"({beyond / len(fixtures):.0%}) fall beyond it and are "
                  f"charged no European load")

    # Squad thinness widens a club's band as well as costing it in congested
    # weeks -- see monte_carlo.team_sigma. Measured against the league, so
    # only clubs genuinely below average carry any of it.
    flags = default_flags(priors=priors, as_of=as_of)

    # How much the fit actually knows about each club (see evidence_share).
    # Ever-present clubs land a couple of tenths of a percent apart purely
    # from which weekend they happened to play on -- that is scheduling
    # jitter, not a real difference in evidence, so anything above the
    # threshold counts as fully known.
    evidence = evidence_share(matches, as_of)
    thin_evidence = []
    for team in config.TEAMS:
        share = evidence.get(team, 0.0)
        if share < config.EVIDENCE_FULL_THRESHOLD:
            flags.setdefault(team, {})["evidence"] = share
            thin_evidence.append((share, team))
    if thin_evidence:
        print("  thin match evidence: " + ", ".join(
            f"{t} ({v:.0%})" for v, t in sorted(thin_evidence)))

    depth_values = np.array([v for v in depth.values() if np.isfinite(v)])
    mean_depth, spread = depth_values.mean(), depth_values.std()
    if spread > 1e-6:
        for team, d in depth.items():
            if not np.isfinite(d):
                continue
            thinness = float(np.clip((mean_depth - d) / (2 * spread), 0.0, 1.0))
            if thinness > 0:
                flags.setdefault(team, {})["thinness"] = thinness

    thin = sorted(((f.get("thinness", 0), t) for t, f in flags.items()), reverse=True)[:3]
    print(f"  thinnest squads: " + ", ".join(f"{t} ({v:.2f})" for v, t in thin if v > 0))
    print(f"simulating {config.N_SIMULATIONS:,} seasons "
          f"({len(flags)} clubs on a widened uncertainty band) ...")
    result = simulate(params, fixtures, flags=flags, played=played)

    table = summarise(result)
    grid = rank_matrix(result)
    _sanity(table, result)

    table = table.merge(audit[["team", "source"]], on="team", how="left")
    table["confidence"] = table["source"].map(
        {"promoted": "lower -- squad-value spread uses an unbacktested elasticity"}
    ).fillna("standard")

    if not write:
        _report(table, audit, wrote=False)
        return table, grid, audit

    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.OUTPUT / "prediction_2026_27.csv", index=False)
    grid.to_csv(config.OUTPUT / "rank_matrix_2026_27.csv")
    audit.to_csv(config.OUTPUT / "strengths_2026_27.csv", index=False)

    # The raw draws, kept rather than discarded. Every summary above is a
    # handful of percentiles taken from these arrays; without them a
    # distribution cannot be replotted, only its summary re-read. Stored as
    # int16 -- nothing here exceeds a season's points or goals.
    np.savez_compressed(
        config.OUTPUT / "simulations.npz",
        teams=np.array(result["teams"]),
        points=result["points"].astype(np.int16),
        goal_diff=result["goal_diff"].astype(np.int16),
        goals_for=result["goals_for"].astype(np.int16),
        ranks=result["ranks"].astype(np.int16),
    )

    # The table week by week, so a season reads as something that unfolds
    # rather than a single closing number.
    track = result["track"]
    pd.concat([
        pd.DataFrame({
            "matchweek": np.repeat(track["matchweeks"], len(result["teams"])),
            "team": np.tile(result["teams"], len(track["matchweeks"])),
            "points_mean": track["points_mean"].ravel(),
            "points_p10": track["points_p10"].ravel(),
            "points_p90": track["points_p90"].ravel(),
            "rank_mean": track["rank_mean"].ravel(),
        })
    ]).to_csv(config.OUTPUT / "trajectory_2026_27.csv", index=False)

    _report(table, audit)
    return table, grid, audit


def _sanity(table: pd.DataFrame, result: dict) -> None:
    assert len(table) == 20, f"expected 20 clubs, got {len(table)}"
    assert (np.sort(result["ranks"], axis=1) == np.arange(1, 21)).all(), "ranks not a permutation"
    assert abs(table["P_title"].sum() - 1.0) < 1e-9, "title probabilities must sum to 1"
    assert abs(table["P_bottom3"].sum() - 3.0) < 1e-9, "relegation probabilities must sum to 3"

    total = result["points"].sum(axis=1)
    assert (total >= 380 * 2).all() and (total <= 380 * 3).all(), "league points total out of range"

    # A season where nobody separates from anybody is a broken model, not a
    # tight league.
    spread = table["exp_points"].max() - table["exp_points"].min()
    assert spread > 15, f"no signal: only {spread:.1f} points between first and last"

    if table["exp_points"].max() > 100:
        print(f"  ! warning: top club projected {table['exp_points'].max():.0f} points, "
              "above the all-time record of 100")


def _report(table: pd.DataFrame, audit: pd.DataFrame, wrote: bool = True) -> None:
    a = audit.set_index("team")
    print(f"\n{'#':>3}  {'club':16s} {'pts':>5} {'90% range':>12} {'title':>7} "
          f"{'top4':>7} {'releg':>7}  {'source':>11}")
    print("-" * 84)
    for i, r in table.iterrows():
        source = a.loc[r["team"], "source"]
        label = source + "*" if source == "promoted" else source
        print(
            f"{i + 1:>3}  {r['team']:16s} {r['exp_points']:5.1f} "
            f"{r['points_p10']:5.0f}-{r['points_p90']:<6.0f} "
            f"{r['P_title']:6.1%} {r['P_top4']:6.1%} {r['P_bottom3']:6.1%}  "
            f"{label:>11}"
        )

    if (a["source"] == "promoted").any():
        print(
            f"\n* promoted clubs: level is measured (real ppg of past promoted clubs), but the "
            f"SPREAD between them relies on SQUAD_VALUE_ELASTICITY={config.SQUAD_VALUE_ELASTICITY} "
            f"(ESTIMATE, not backtested -- no historical squad-value data exists to calibrate it "
            f"against). Treat rank/points for promoted clubs as lower-confidence than fitted clubs."
        )

    moved = audit[audit["manager_delta"].abs() > 1e-9].copy()
    if len(moved):
        moved = moved.reindex(moved["manager_delta"].abs().sort_values(ascending=False).index)
        print("\nmanager prior, largest effects:")
        for r in moved.head(6).itertuples(index=False):
            print(f"  {r.team:16s} {r.manager:18s} prior {r.prior_ppg:.2f} ppg "
                  f"(conf {r.prior_confidence:.2f})  ->  atk {r.manager_delta_atk:+.3f} "
                  f"def {r.manager_delta_def:+.3f}")

    print(f"\nwritten to {config.OUTPUT}" if wrote
          else "\n(weekly run -- output/ left untouched, logged by src/weekly.py)")


if __name__ == "__main__":
    main()
