"""
Calibrate SQUAD_VALUE_ELASTICITY against the six real promoted clubs whose
historical squads the user collected (Ipswich/Leicester/Southampton at
August 2024, Burnley/Leeds/Sunderland at August 2025).

    python -m src.squad_elasticity

What this tests is narrower than "does squad value predict points" -- the
level (how well promoted clubs do on average) is already measured elsewhere
from real results (run_pipeline.promoted_baseline). This asks only: within a
promoted-season cohort of three, does the SPREAD in squad value predict the
spread in how they actually performed, and by how much?

Method, mirroring run_pipeline.build_strengths()'s promoted-club logic
exactly so the fitted elasticity means the same thing it will be used for:
  1. For each season, fit Dixon-Coles on seasons strictly before it (same
     fold structure as backtest.py) and measure the real ppg-to-strength
     slope from that fit (ppg_to_strength_slope), so squad value and match
     results land in the same units.
  2. Each club's real combined strength = (actual ppg - league average) *
     slope, i.e. exactly what it earned on the pitch that season.
  3. Within its 3-club cohort, demean both squad value (log) and combined
     strength -- squad_to_strength() only ever explains the spread around a
     level that promoted_baseline() supplies separately, so fitting on raw
     values here would answer a question nobody is asking.
  4. Pool both cohorts and fit ELASTICITY by ordinary least squares through
     the origin (no intercept -- zero log-value gap must mean zero strength
     gap, by construction).

Six points cannot produce a precise number; what this CAN do is say whether
the current default (0.36) is in a defensible range or wildly off, and give
a specific value with an honest sample-size caveat attached.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src.dixon_coles import fit
from src.run_pipeline import load_matches, ppg_to_strength_slope

_CLUB_NAME = {
    "Southampton FC": "Southampton",
    "Ipswich Town": "Ipswich Town",
    "Leicester City": "Leicester City",
    "Sunderland AFC": "Sunderland",
    "Leeds United": "Leeds United",
    "Burnley FC": "Burnley",
    "Sheffield United": "Sheffield United",
    "Luton Town": "Luton Town",
    "Nottingham Forest": "Nott'ham Forest",
    "Fulham FC": "Fulham",
    "AFC Bournemouth": "Bournemouth",
    "West Bromwich Albion": "West Brom",
}

# Cohorts with real match data in matches_football_data.csv (2023-2024 onward):
# a slope can be measured -- from a prior season for the two original cohorts,
# from the season itself for 2023-2024 (it's the earliest season on file, so
# there is no "prior" fold; fitting on its own matches is fine here because
# this is retrospective calibration, not out-of-sample forecasting).
_COHORTS = {
    "2024-2025": ["Southampton FC", "Ipswich Town", "Leicester City"],
    "2025-2026": ["Sunderland AFC", "Leeds United", "Burnley FC"],
    "2023-2024": ["Burnley FC", "Sheffield United", "Luton Town"],
}

# Cohorts with NO match data available at all (older than 2023-2024). Real
# final points come from a verified external source (Wikipedia / Sky Sports,
# cross-checked); no slope can be measured, so points-to-strength conversion
# reuses the average of the two originally-measured slopes (~0.94) as a
# labelled approximation, not a fit.
_PROXY_SLOPE = (0.931 + 0.961) / 2
_EXTERNAL_COHORTS = {
    "2022-2023": {
        "Fulham FC": (52, 38),
        "AFC Bournemouth": (39, 38),
        "Nottingham Forest": (38, 38),
    },
    "2020-2021": {
        "Leeds United": (59, 38),
        "Fulham FC": (28, 38),
        "West Bromwich Albion": (26, 38),
    },
}


def squad_values() -> dict[tuple[str, str], float]:
    """Top-18 raw squad value per (club, season), from the user-supplied historical squads.

    Deliberately raw, not adaptation-discounted: adaptation_factor() needs
    each player's EPL appearance count AS OF that historical date, which
    isn't available for a past snapshot without re-fetching a point-in-time
    FPL history this project doesn't have. Using the plain top-18 sum is a
    real simplification -- noted, not hidden -- and it's still a fair test
    of the elasticity's functional form.

    Keyed by (club, season) rather than club alone: the second batch of
    historical squads has some clubs (Fulham) appearing in more than one
    season with a different squad each time.
    """
    rows = []
    for fname in ("promoted_historical_squads_parsed.json", "promoted_historical_squads_2_parsed.json"):
        path = config.RAW / fname
        if path.exists():
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    df["season"] = df["season"].apply(lambda s: s if "-" in s else f"20{s[:2]}-20{s[3:]}")
    out = {}
    for (club, season), sub in df.groupby(["club", "season"]):
        top18 = sub.nlargest(config.SQUAD_DEPTH_COUNT, "market_value_m")
        out[(club, season)] = float(top18["market_value_m"].sum())
    return out


def real_ppg(matches: pd.DataFrame, team: str, season: str) -> tuple[float, int]:
    d = matches[(matches["season"] == season) & ((matches["home_team"] == team) | (matches["away_team"] == team))]
    pts = 0
    for r in d.itertuples(index=False):
        home = r.home_team == team
        gf, ga = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
        pts += config.POINTS_WIN if gf > ga else (config.POINTS_DRAW if gf == ga else config.POINTS_LOSS)
    return (pts / len(d) if len(d) else float("nan")), len(d)


def main() -> None:
    matches = load_matches()
    all_seasons = sorted(matches["season"].unique())
    values = squad_values()

    xs, ys, detail = [], [], []
    for season, clubs in _COHORTS.items():
        idx = all_seasons.index(season)
        # 2023-2024 is the earliest season on file -- no strictly-prior fold
        # exists, so fit on the season itself. Fine for retrospective
        # calibration (not used for forecasting), not fine for backtest.py.
        train = matches[matches["season"].isin(all_seasons[:idx])] if idx > 0 else matches[matches["season"] == season]
        as_of = pd.Timestamp(matches.loc[matches["season"] == season, "date"].min())
        fitted = fit(train, target="goals", reference_date=as_of)
        slope = ppg_to_strength_slope(train, fitted, as_of)

        cohort_val, cohort_combined = [], []
        for club in clubs:
            team = _CLUB_NAME[club]
            ppg, n = real_ppg(matches, team, season)
            combined = (ppg - config.LEAGUE_AVG_PPG) * slope
            val = values[(club, season)]
            cohort_val.append(val)
            cohort_combined.append(combined)
            detail.append((season, team, val, ppg, n, combined, "measured"))

        log_val = np.log(cohort_val)
        x = log_val - log_val.mean()
        y = np.array(cohort_combined) - np.mean(cohort_combined)
        xs.extend(x)
        ys.extend(y)
        print(f"{season}  slope={slope:.3f}  " + "  ".join(
            f"{_CLUB_NAME[c]}={v:.0f}m" for c, v in zip(clubs, cohort_val)))

    # Older cohorts: no match data at all, so the slope is a labelled proxy
    # (average of the two measured slopes above), not a fit. Kept in a
    # separate pool from the measured cohorts so a weaker approximation
    # can't quietly dominate the "real" result.
    xs_proxy, ys_proxy, detail_proxy = [], [], []
    for season, clubs in _EXTERNAL_COHORTS.items():
        cohort_val, cohort_combined = [], []
        for club, (pts, n) in clubs.items():
            ppg = pts / n
            combined = (ppg - config.LEAGUE_AVG_PPG) * _PROXY_SLOPE
            val = values[(club, season)]
            cohort_val.append(val)
            cohort_combined.append(combined)
            detail_proxy.append((season, _CLUB_NAME[club], val, ppg, n, combined, "proxy slope"))

        log_val = np.log(cohort_val)
        x = log_val - log_val.mean()
        y = np.array(cohort_combined) - np.mean(cohort_combined)
        xs_proxy.extend(x)
        ys_proxy.extend(y)
        print(f"{season}  slope={_PROXY_SLOPE:.3f} (proxy)  " + "  ".join(
            f"{_CLUB_NAME[c]}={v:.0f}m" for c, v in zip(clubs, cohort_val)))

    all_cohorts_seasons = list(_COHORTS) + list(_EXTERNAL_COHORTS)
    xs, ys, detail = xs + xs_proxy, ys + ys_proxy, detail + detail_proxy

    xs, ys = np.array(xs), np.array(ys)
    elasticity = float(np.sum(xs * ys) / np.sum(xs * xs))
    residual = ys - elasticity * xs
    ss_res, ss_tot = float(np.sum(residual ** 2)), float(np.sum(ys ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # A pooled fit can look confident while being carried entirely by one
    # cohort. With five cohorts of three, this leave-one-out check is what
    # tells a real pattern from one big-spread cohort dominating the sum.
    print("\nper-cohort check (does each season agree on its own?):")
    per_cohort_sign = {}
    for season in all_cohorts_seasons:
        mask = [d[0] == season for d in detail]
        cx = xs[np.array(mask)]
        cy = ys[np.array(mask)]
        e = float(np.sum(cx * cy) / np.sum(cx * cx)) if np.sum(cx * cx) > 0 else float("nan")
        per_cohort_sign[season] = e > 0
        tag = " (proxy slope)" if season in _EXTERNAL_COHORTS else ""
        print(f"  {season}: elasticity from this cohort alone = {e:+.2f}{tag}")

    print(f"\n{'season':10s}{'club':16s}{'value':>8}{'real ppg':>10}{'matches':>9}{'combined':>10}  basis")
    for season, team, val, ppg, n, combined, basis in detail:
        print(f"{season:10s}{team:16s}{val:8.0f}{ppg:10.2f}{n:9d}{combined:10.3f}  {basis}")

    n_clubs = len(detail)
    print(f"\npooled fit: SQUAD_VALUE_ELASTICITY = {elasticity:.3f}  "
          f"(current default: {config.SQUAD_VALUE_ELASTICITY}, R2={r2:.2f}, n={n_clubs})")

    n_positive = sum(per_cohort_sign.values())
    signs_agree = n_positive == 0 or n_positive == len(per_cohort_sign)
    if not signs_agree:
        detail_str = ", ".join(f"{s}={'+' if v else '-'}" for s, v in per_cohort_sign.items())
        print(
            f"\nNOT adopting this number. Cohorts disagree on the SIGN of the "
            f"relationship ({detail_str}) -- some seasons say higher squad value "
            "within a promoted trio predicts a better finish, others say the "
            "opposite. A pooled R2 that looks strong can still be an artifact of "
            "whichever cohort has the biggest spread, not genuine cross-season "
            "agreement. SQUAD_VALUE_ELASTICITY stays at its current default; "
            "treat this as evidence the parameter needs a methodology this "
            "project doesn't have yet (point-in-time squad data plus a longer "
            "run of real seasons), not as a new number to plug in."
        )
    else:
        print(
            f"\nAll {len(per_cohort_sign)} cohorts agree on direction. Still only "
            f"{n_clubs} clubs total -- treat {elasticity:.2f} as a signal the "
            f"current default's ballpark is defensible, not as a precise "
            "replacement value, and update the config default only if this "
            "holds up with even more seasons."
        )


if __name__ == "__main__":
    main()
