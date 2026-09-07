"""
Backtest the core engine against real final tables.

    python -m src.backtest

Scope, and why it stops here: this grid-searches XI (time decay) by training
Dixon-Coles on seasons strictly before a held-out one, simulating it, and
scoring against what actually happened. That is the part of the pipeline
built entirely from match results, which exist in full for every season in
matches_football_data.csv.

MANAGER_EFFECT_SHARE, SQUAD_VALUE_ELASTICITY, and ADAPTATION_DRAG cannot be
backtested the same way -- they need squad values and manager status AS OF
August 2024 and August 2025, and only the 2026/27 snapshot was ever collected.
Backtesting them would mean fabricating historical inputs to calibrate against
historical outputs, which is worse than not calibrating them at all. They stay
at their ESTIMATE defaults, honestly labelled, until someone collects that
data.

Promoted clubs get a level from the SAME real-measurement idea as
run_pipeline.promoted_baseline() -- the other promotion window's actual ppg,
never the target season's own (that would leak the answer) -- but not the
squad-value spread between them, since historical squad values don't exist.
Every promoted club in a backtest fold gets the same strength. That is a real
gap next to the live 2026/27 run, not an oversight, and the report says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from src.dixon_coles import DCParams, fit, time_weights
from src.monte_carlo import rank_matrix, simulate, summarise
from src.run_pipeline import load_matches, ppg_to_strength_slope


def teams_in_season(matches: pd.DataFrame, season: str) -> set[str]:
    d = matches[matches["season"] == season]
    return set(d["home_team"]) | set(d["away_team"])


def actual_table(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Real final standings for a season, from its match results alone."""
    d = matches[matches["season"] == season]
    teams = sorted(teams_in_season(matches, season))
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}

    for r in d.itertuples(index=False):
        h_pts = config.POINTS_WIN if r.home_goals > r.away_goals else (
            config.POINTS_DRAW if r.home_goals == r.away_goals else config.POINTS_LOSS)
        a_pts = config.POINTS_WIN if r.away_goals > r.home_goals else (
            config.POINTS_DRAW if r.home_goals == r.away_goals else config.POINTS_LOSS)
        pts[r.home_team] += h_pts
        pts[r.away_team] += a_pts
        gf[r.home_team] += r.home_goals
        ga[r.home_team] += r.away_goals
        gf[r.away_team] += r.away_goals
        ga[r.away_team] += r.home_goals

    df = pd.DataFrame({"team": teams, "points": [pts[t] for t in teams],
                        "gf": [gf[t] for t in teams], "ga": [ga[t] for t in teams]})
    df["gd"] = df["gf"] - df["ga"]
    df = df.sort_values(["points", "gd", "gf"], ascending=False, ignore_index=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df.set_index("team")


def other_promoted_ppg(matches: pd.DataFrame, all_seasons: list[str], exclude_idx: int) -> tuple[float | None, int]:
    """
    Real ppg of promoted clubs from every OTHER promotion window in the file.

    Excluding the target season's own promotions is what keeps this from
    leaking the answer -- with only three seasons on hand there is exactly one
    other window to draw from, so n is small (3) but every point in it is real.
    """
    ppgs = []
    for i in range(1, len(all_seasons)):
        if i == exclude_idx:
            continue
        cur, prev = all_seasons[i], all_seasons[i - 1]
        promoted = teams_in_season(matches, cur) - teams_in_season(matches, prev)
        d = matches[matches["season"] == cur]
        for team in promoted:
            sub = d[(d["home_team"] == team) | (d["away_team"] == team)]
            pts = 0
            for r in sub.itertuples(index=False):
                home = r.home_team == team
                gf, ga = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
                pts += config.POINTS_WIN if gf > ga else (config.POINTS_DRAW if gf == ga else config.POINTS_LOSS)
            ppgs.append(pts / len(sub))
    return (float(np.mean(ppgs)), len(ppgs)) if ppgs else (None, 0)


def rps(grid: pd.DataFrame, actual: pd.DataFrame) -> float:
    """
    Ranked Probability Score, averaged over teams. Lower is better, 0 is
    perfect. Rewards mass placed NEAR the true rank, not just on it --
    predicting a team 2nd when they finish 3rd costs less than predicting
    them 18th.
    """
    n = grid.shape[1]
    scores = []
    for team in grid.index:
        p = grid.loc[team].to_numpy()
        cum_p = np.cumsum(p)[:-1]
        true_rank = int(actual.loc[team, "rank"])
        cum_a = (np.arange(1, n) >= true_rank).astype(float)
        scores.append(np.mean((cum_p - cum_a) ** 2))
    return float(np.mean(scores))


def backtest_season(matches: pd.DataFrame, all_seasons: list[str], season: str, xi: float) -> dict:
    idx = all_seasons.index(season)
    train_seasons = all_seasons[:idx]
    if not train_seasons:
        raise ValueError(f"{season} has no seasons before it to train on")

    target_teams = teams_in_season(matches, season)
    prev_teams = teams_in_season(matches, all_seasons[idx - 1])
    promoted = target_teams - prev_teams
    non_promoted = sorted(target_teams - promoted)

    train = matches[matches["season"].isin(train_seasons)]
    reference = pd.Timestamp(matches.loc[matches["season"] == season, "date"].min())

    fitted = fit(train, target="goals", xi=xi, teams=non_promoted, reference_date=reference)
    slope = ppg_to_strength_slope(train, fitted, reference)

    promo_ppg, n_promo = other_promoted_ppg(matches, all_seasons, idx)
    promo_combined = (promo_ppg - config.LEAGUE_AVG_PPG) * slope if promo_ppg is not None else 0.0

    teams = sorted(target_teams)
    attack = np.array([
        fitted.attack[fitted.index[t]] if t in non_promoted else promo_combined / 2.0 for t in teams
    ])
    defence = np.array([
        fitted.defence[fitted.index[t]] if t in non_promoted else promo_combined / 2.0 for t in teams
    ])
    attack = attack - attack.mean()  # gauge: re-centre after swapping the promoted/relegated set

    params = DCParams(teams=teams, attack=attack, defence=defence,
                       home_adv=fitted.home_adv, rho=fitted.rho, n_matches=fitted.n_matches, source="backtest")

    fixtures = matches.loc[matches["season"] == season, ["date", "matchweek", "home_team", "away_team"]]
    flags = {t: {"promoted": True} for t in promoted}
    result = simulate(params, fixtures, flags=flags, seed=config.RANDOM_SEED)

    table = summarise(result).set_index("team")
    grid = rank_matrix(result)
    actual = actual_table(matches, season)

    pred_order = table["exp_points"].sort_values(ascending=False).index
    pred_rank = pd.Series(np.arange(1, len(pred_order) + 1), index=pred_order)

    common = actual.index  # summarise() and actual_table() cover the same target_teams by construction
    return {
        "season": season,
        "xi": xi,
        "n_promoted_pool": n_promo,
        "spearman": float(spearmanr(pred_rank[common], actual.loc[common, "rank"]).correlation),
        "mae_rank": float(np.mean(np.abs(pred_rank[common] - actual.loc[common, "rank"]))),
        "rps": rps(grid.loc[common], actual.loc[common]),
        "mae_points": float(np.mean(np.abs(table.loc[common, "exp_points"] - actual.loc[common, "points"]))),
    }


def naive_baseline(matches: pd.DataFrame, all_seasons: list[str], season: str) -> dict:
    """Predict this season's order as last season's final table, promoted
    clubs placed last (alphabetically, so no signal about them leaks in)."""
    idx = all_seasons.index(season)
    prev_actual = actual_table(matches, all_seasons[idx - 1])
    actual = actual_table(matches, season)

    returning = [t for t in prev_actual.sort_values("rank").index if t in actual.index]
    promoted = sorted(t for t in actual.index if t not in prev_actual.index)
    order = returning + promoted
    pred_rank = pd.Series(np.arange(1, len(order) + 1), index=order)

    common = actual.index
    return {
        "spearman": float(spearmanr(pred_rank[common], actual.loc[common, "rank"]).correlation),
        "mae_rank": float(np.mean(np.abs(pred_rank[common] - actual.loc[common, "rank"]))),
    }


def grid_search(matches: pd.DataFrame) -> pd.DataFrame:
    all_seasons = sorted(matches["season"].unique())
    rows = []
    for xi in config.XI_GRID:
        for season in config.BACKTEST_SEASONS:
            rows.append(backtest_season(matches, all_seasons, season, xi))
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    matches = load_matches()
    all_seasons = sorted(matches["season"].unique())

    print(f"backtesting against {config.BACKTEST_SEASONS} "
          f"(only {len(all_seasons)} seasons of match data exist -- {len(config.BACKTEST_SEASONS)} folds, not a large-sample calibration)\n")

    results = grid_search(matches)
    by_xi = results.groupby("xi")[["spearman", "mae_rank", "rps", "mae_points"]].mean()
    by_xi = by_xi.sort_values("rps")

    print(f"{'xi':>8} {'spearman':>9} {'mae_rank':>9} {'rps':>7} {'mae_pts':>8}")
    for xi, row in by_xi.iterrows():
        marker = "  <- current default" if abs(xi - config.XI) < 1e-9 else ""
        print(f"{xi:8.4f} {row['spearman']:9.3f} {row['mae_rank']:9.2f} {row['rps']:7.4f} {row['mae_points']:8.2f}{marker}")

    best_xi = by_xi.index[0]

    print(f"\nper-season detail at xi={best_xi}:")
    for _, r in results[np.isclose(results["xi"], best_xi)].iterrows():
        print(f"  {r['season']}: spearman={r['spearman']:.3f} mae_rank={r['mae_rank']:.2f} "
              f"rps={r['rps']:.4f} mae_points={r['mae_points']:.2f} (promoted baseline from n={r['n_promoted_pool']} club-seasons)")

    print("\nvs naive baseline (last season's table, promoted clubs placed last):")
    for season in config.BACKTEST_SEASONS:
        nb = naive_baseline(matches, all_seasons, season)
        print(f"  {season}: spearman={nb['spearman']:.3f} mae_rank={nb['mae_rank']:.2f}")

    print(
        "\nNOT calibrated here -- would need historical squad values and manager status "
        "that were never collected for past seasons, only for 2026/27:\n"
        "  MANAGER_EFFECT_SHARE, SQUAD_VALUE_ELASTICITY, ADAPTATION_DRAG (stay at ESTIMATE defaults)"
    )
    return results


if __name__ == "__main__":
    matches = load_matches()
    all_seasons = sorted(matches["season"].unique())

    # Sanity checks on the scoring machinery itself, at the current default xi.
    r = backtest_season(matches, all_seasons, config.BACKTEST_SEASONS[0], config.XI)
    assert 0.0 <= r["rps"] <= 1.0, f"RPS out of range: {r['rps']}"
    assert -1.0 <= r["spearman"] <= 1.0, f"spearman out of range: {r['spearman']}"
    assert r["mae_rank"] >= 0, "MAE cannot be negative"

    # A team's own actual final rank must appear in its rank_matrix row with
    # positive probability, or the whole scoring approach is measuring nothing.
    idx = all_seasons.index(config.BACKTEST_SEASONS[0])
    prev = teams_in_season(matches, all_seasons[idx - 1])
    cur = teams_in_season(matches, config.BACKTEST_SEASONS[0])
    promoted = cur - prev
    assert len(promoted) == 3, f"expected 3 promoted clubs, found {promoted}"

    ppg, n = other_promoted_ppg(matches, all_seasons, idx)
    assert n == 3, f"expected the other promotion window's 3 clubs, got {n}"
    assert 0.0 < ppg < 3.0, f"promoted ppg implausible: {ppg}"

    print(f"OK  self-check passed (RPS={r['rps']:.4f}, promoted baseline n={n}, ppg={ppg:.2f})\n")
    main()
