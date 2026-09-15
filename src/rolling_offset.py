"""
Rolling out-of-sample test of the Championship-to-PL offset.

    python -m src.rolling_offset

The two-fold backtest can only ask "did the offset help in 2024-25 and
2025-26?" -- six club-seasons, one draw each from the tails. This asks the
question with power: for every year T from 2000, fit the offset on promotion
windows whose Premier League side is T or earlier, predict the clubs promoted
into T+1 without ever seeing them, score, advance. 25 out-of-sample seasons
(2000-2024), 75 scored clubs, from championship.big_pairs().

Three models per fold, so the interaction the pooled fit was hiding gets its
own test rather than an assumption:

    M0  population mean       pl_ppg = mean of training clubs
    M1  pooled offset         pl_ppg = a + b * champ_gd
    M2  parachute interaction separate (a, b) for clubs that were in the top
                              flight within the three seasons before their
                              promotion season (the parachute-payment window)
                              and for clubs that were not

Two scores. Squared error on points per game, the quantity the offset predicts.
And Brier on relegation, the quantity the whole argument is about: P(relegated)
is Phi((threshold - predicted) / sigma), with sigma the residual sd of the
training fit and the threshold the mean 18th-place ppg over the training
seasons -- nothing from the scored season leaks into its own prediction.

Brier is judged against RATE, the training relegation rate of promoted clubs
(climatology). Promoted clubs go down about 45% of the time, so a perfectly
calibrated constant already scores ~0.247; the absolute 0.25 coin-flip line is
not the bar here, the skill score against RATE is.

Relegation truth is bottom three of the tier-1 table by points then goal
difference. The three 22-team seasons (1992-95) relegated four; treated as
three here, and said so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import championship

_FIRST_SCORED = 2000       # first out-of-sample season; 8 training windows behind it
_PARACHUTE_YEARS = 3       # top-flight presence within this many seasons before promotion


def _tier1_tables(e: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """{season: table sorted by points, gd} for tier 1 -- relegation truth."""
    out = {}
    for season in sorted(e.loc[e["tier"] == 1, "Season"].unique()):
        t = championship._season_stats(e, int(season), 1)
        t = t.sort_values(["ppg", "gd"], ascending=False).reset_index(drop=True)
        t["rank"] = np.arange(1, len(t) + 1)
        out[int(season)] = t
    return out


def _pairs_with_flags(e: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """championship.big_pairs() plus a parachute flag and a relegated flag."""
    p = championship.big_pairs()
    tables = _tier1_tables(e)
    top = {s: set(t["club"]) for s, t in tables.items()}

    def parachute(club: str, season: int) -> bool:
        return any(club in top.get(season - k, set()) for k in range(1, _PARACHUTE_YEARS + 1))

    def relegated(club: str, pl_season: int) -> bool:
        t = tables[pl_season]
        return bool(t.loc[t["club"] == club, "rank"].iloc[0] > len(t) - 3)

    p["parachute"] = [parachute(r.club, r.season) for r in p.itertuples(index=False)]
    p["relegated"] = [relegated(r.club, r.season + 1) for r in p.itertuples(index=False)]
    p["pl_season"] = p["season"] + 1
    return p, tables


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(a, b, residual sd) with b clamped to [0, 1], as in championship.fit."""
    if len(x) < 3 or x.std() < 1e-9:
        return float(y.mean()), 0.0, float(y.std(ddof=1)) if len(y) > 1 else 0.3
    b_raw, _ = np.polyfit(x, y, 1)
    b = float(np.clip(b_raw, 0.0, 1.0))
    a = float(y.mean() - b * x.mean())
    resid = y - (a + b * x)
    return a, b, float(resid.std(ddof=1))


def rolling() -> pd.DataFrame:
    """One row per (fold, club, model): prediction, actual, Brier inputs."""
    e = championship._eng()
    if e is None:
        raise FileNotFoundError("data/raw/england_all_tiers.csv missing")
    pairs, tables = _pairs_with_flags(e)
    last = int(pairs["pl_season"].max())

    rows = []
    for T in range(_FIRST_SCORED - 1, last):
        train = pairs[pairs["pl_season"] <= T]
        test = pairs[pairs["pl_season"] == T + 1]
        if train.empty or test.empty:
            continue

        # Relegation threshold from training seasons only.
        thr = float(np.mean([tables[s].iloc[len(tables[s]) - 3]["ppg"]
                             for s in train["pl_season"].unique()]))

        # M0: population mean
        m0 = (float(train["pl_ppg"].mean()), 0.0, float(train["pl_ppg"].std(ddof=1)))
        # M1: pooled line on GD
        m1 = _fit_line(train["champ_gd"].to_numpy(), train["pl_ppg"].to_numpy())
        # M2: one line per parachute group
        m2 = {}
        for flag in (True, False):
            g = train[train["parachute"] == flag]
            m2[flag] = _fit_line(g["champ_gd"].to_numpy(), g["pl_ppg"].to_numpy()) if len(g) >= 3 else m1

        assert train["pl_season"].max() < T + 1, "leak: test season inside training"
        rate = float(train["relegated"].mean())

        for r in test.itertuples(index=False):
            preds = {
                "M0": (m0[0], m0[2]),
                "M1": (m1[0] + m1[1] * r.champ_gd, m1[2]),
                "M2": (m2[r.parachute][0] + m2[r.parachute][1] * r.champ_gd, m2[r.parachute][2]),
                "RATE": (m0[0], None),
            }
            for model, (mu, sd) in preds.items():
                p_rel = rate if sd is None else float(norm.cdf((thr - mu) / max(sd, 1e-6)))
                rows.append({
                    "fold": T + 1, "club": r.club, "parachute": r.parachute,
                    "model": model, "pred_ppg": mu, "actual_ppg": r.pl_ppg,
                    "p_relegated": p_rel, "relegated": int(r.relegated),
                    "sq_err": (mu - r.pl_ppg) ** 2,
                    "brier": (p_rel - r.relegated) ** 2,
                })
    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-model aggregate, per-fold wins) over the rolling results."""
    agg = df.groupby("model").agg(
        folds=("fold", "nunique"), clubs=("club", "size"),
        mse_ppg=("sq_err", "mean"), brier=("brier", "mean"),
    )
    agg["brier_skill"] = 1 - agg["brier"] / agg.loc["RATE", "brier"]
    agg = agg.round(4)

    by_fold = (df[df.model != "RATE"].groupby(["fold", "model"])[["sq_err", "brier"]]
               .mean().unstack("model"))
    wins = pd.DataFrame({
        "mse_wins": by_fold["sq_err"].idxmin(axis=1).value_counts(),
        "brier_wins": by_fold["brier"].idxmin(axis=1).value_counts(),
    }).fillna(0).astype(int)
    return agg, wins


def main() -> None:
    df = rolling()
    agg, wins = score(df)
    n_folds = df["fold"].nunique()

    print(f"rolling out-of-sample, {n_folds} folds ({df['fold'].min()}-{df['fold'].max()}), "
          f"{df[df.model == 'M0'].shape[0]} promoted clubs scored\n")
    labels = {"M0": "population mean", "M1": "pooled GD offset",
              "M2": "parachute-split offset", "RATE": "training relegation rate (climatology)"}
    print("model  folds  clubs   MSE ppg    Brier    skill vs RATE")
    for m in ("RATE", "M0", "M1", "M2"):
        r = agg.loc[m]
        print(f"{m:5s} {int(r.folds):6d} {int(r.clubs):6d}   {r.mse_ppg:8.4f}   {r.brier:7.4f}   "
              f"{r.brier_skill:+7.3f}     {labels[m]}")

    print("\nfolds won (lowest error that season):")
    for m in ("M0", "M1", "M2"):
        w = wins.reindex(["M0", "M1", "M2"]).fillna(0).astype(int)
        print(f"  {m}: MSE {int(w.loc[m, 'mse_wins']):2d}   Brier {int(w.loc[m, 'brier_wins']):2d}")

    # Where does M2 earn its keep -- on which subgroup?
    print("\nby subgroup (mean Brier):")
    sub = df.groupby(["parachute", "model"])["brier"].mean().unstack("model").round(4)
    sub.index = ["without parachute", "with parachute"]
    print(sub.to_string())

    # Parachute payments only became large after 2012; test the interaction
    # where its mechanism actually exists.
    late = df[df.fold >= 2013]
    print(f"\nfolds {late.fold.min()}-{late.fold.max()} only ({late.fold.nunique()} folds, "
          f"{late[late.model == 'M0'].shape[0]} clubs):")
    print(late.groupby("model")[["sq_err", "brier"]].mean().round(4).to_string())

    print("\nper fold (Brier):")
    pf = df.groupby(["fold", "model"])["brier"].mean().unstack("model").round(3)
    pf["relegated"] = df[df.model == "M0"].groupby("fold")["relegated"].sum()
    print(pf[["M0", "M1", "M2", "RATE", "relegated"]].to_string())

    # The last two folds are the ones the 2-fold backtest could see.
    tail = df[df.fold >= df.fold.max() - 1].groupby("model")[["sq_err", "brier"]].mean().round(4)
    print(f"\nlast two folds only ({df.fold.max()-1}, {df.fold.max()}), for comparison with backtest.py:")
    print(tail.to_string())

    # Sanity: enough folds, probabilities well-formed, climatology near its algebraic value.
    assert n_folds >= 20, f"only {n_folds} folds -- not the power this test is for"
    assert df["p_relegated"].between(0, 1).all()
    rate = df[df.model == "M0"]["relegated"].mean()
    assert abs(agg.loc["RATE", "brier"] - rate * (1 - rate)) < 0.02, "climatology Brier off its p(1-p)"
    best = agg["brier"].idxmin()
    print(f"\nOK  {n_folds} folds scored; lowest Brier: {best}")


if __name__ == "__main__":
    main()
