"""
How strong is each club's prior, measured in matches?

    python -m src.prior_strength

The peer's metric from the LinkedIn thread: re-run the model conditioned on the
first N matchdays for N = 0, 1, 2, ... and see how many it takes to move a club's
relegation probability by ten percentage points. That number is the prior's
strength in units of real matches -- comparable across clubs, and it separates a
prior that updates from a wall that doesn't.

Only three matchdays have been played, so the direct answer exists for clubs
that cross ten points inside that window; for the rest the per-matchday slope is
reported and a projection given, clearly marked as extrapolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import run_pipeline as rp
from src.live_results import fetch_played

_TARGET = 0.10  # the ten-point shift the metric is defined around


def _series() -> pd.DataFrame:
    """P_bottom3 for every club at N = 0 .. matchdays played."""
    played = fetch_played(use_cache=True)
    max_n = int(played["matchweek"].max()) if len(played) else 0

    rows = []
    for n in range(max_n + 1):
        subset = None if n == 0 else played[played["matchweek"] <= n]
        print(f"\n{'='*60}\nconditioning on {n} matchday(s)\n{'='*60}")
        table, _, _ = rp.main(played=subset, write=False)
        for r in table.itertuples(index=False):
            rows.append({"n": n, "team": r.team,
                         "P_bottom3": r.P_bottom3, "P_title": r.P_title,
                         "exp_points": r.exp_points})
    return pd.DataFrame(rows)


def _matchdays_to_target(g: pd.DataFrame) -> tuple[float, bool, float]:
    """
    (matchdays to a ten-point relegation shift, was_it_reached, slope_per_md).

    Reached: linear-interpolate between the two matchdays that straddle the
    threshold. Not reached: extrapolate off the average per-matchday slope.
    """
    g = g.sort_values("n")
    base = g[g["n"] == 0]["P_bottom3"].iloc[0]
    shift = (g["P_bottom3"] - base).abs()
    n = g["n"].to_numpy()

    slope = float(np.polyfit(n, g["P_bottom3"].to_numpy(), 1)[0]) if len(g) > 1 else 0.0

    crossed = shift[shift >= _TARGET]
    if len(crossed):
        i = shift.to_numpy().argmax() if shift.iloc[-1] < _TARGET else np.argmax(shift.to_numpy() >= _TARGET)
        if i == 0:
            return 0.0, True, slope
        lo, hi = shift.iloc[i - 1], shift.iloc[i]
        frac = (_TARGET - lo) / (hi - lo) if hi > lo else 0.0
        return float(n[i - 1] + frac * (n[i] - n[i - 1])), True, slope

    projected = _TARGET / abs(slope) if abs(slope) > 1e-6 else float("inf")
    return projected, False, slope


def _selfcheck() -> None:
    """The metric math, on synthetic clubs, before spending 4 pipeline runs."""
    straddle = pd.DataFrame({"n": [0, 1, 2, 3], "P_bottom3": [0.50, 0.55, 0.60, 0.65]})
    md, reached, _ = _matchdays_to_target(straddle)
    assert reached and abs(md - 2.0) < 1e-6, (md, reached)

    wall = pd.DataFrame({"n": [0, 1, 2, 3], "P_bottom3": [0.02, 0.02, 0.019, 0.021]})
    _, reached, _ = _matchdays_to_target(wall)
    assert not reached, "a flat series must not read as a threshold crossing"


def main() -> None:
    _selfcheck()
    df = _series()
    max_n = int(df["n"].max())

    print(f"\n\n{'#'*70}\nPrior strength: matchdays to a {_TARGET:.0%} relegation-probability shift")
    print(f"(measured over {max_n} matchdays actually played)\n{'#'*70}\n")

    out = []
    for team, g in df.groupby("team"):
        md, reached, slope = _matchdays_to_target(g)
        base = g[g["n"] == 0]["P_bottom3"].iloc[0]
        now = g[g["n"] == max_n]["P_bottom3"].iloc[0]
        promoted = team in config.PROMOTED
        out.append({
            "team": team, "promoted": promoted,
            "releg_start": base, "releg_now": now,
            "shift_so_far": now - base,
            "matchdays_to_10pt": md, "reached": reached,
            "slope_per_md": slope,
        })

    res = pd.DataFrame(out).sort_values("matchdays_to_10pt")

    def fmt(r):
        if r.reached:
            return f"{r.matchdays_to_10pt:.1f}"
        # A projection past a full season, or off a slope under half a point a
        # matchday, is not a number -- it is the model saying "this prior does
        # not move". Report it as a wall rather than a spurious 3000.
        if not np.isfinite(r.matchdays_to_10pt) or r.matchdays_to_10pt > 38 or abs(r.slope_per_md) < 0.005:
            return "wall"
        return f"~{r.matchdays_to_10pt:.0f} (proj.)"
    print(f"{'club':16s} {'':3s} {'releg 0->now':>14s} {'md to 10pt':>12s}  {'per-md':>7s}")
    for r in res.itertuples(index=False):
        tag = "P" if r.promoted else " "
        print(f"{r.team:16s} {tag:3s} {r.releg_start:5.1%} -> {r.releg_now:5.1%} "
              f"{fmt(r):>12s}  {r.slope_per_md:+7.1%}")

    prom = res[res["promoted"]]
    fit = res[~res["promoted"]]
    print(f"\npromoted clubs: median {prom['matchdays_to_10pt'].median():.1f} matchdays to a 10-point move")
    fit_reached = fit[fit["reached"]]
    if len(fit_reached):
        print(f"fitted clubs:   median {fit_reached['matchdays_to_10pt'].median():.1f} (of those that reached it)")
    else:
        print(f"fitted clubs:   none moved 10 points in {max_n} matchdays -- the prior is a wall on this timescale")

    res.to_csv(config.OUTPUT / "prior_strength.csv", index=False)
    print(f"\nwritten to {config.OUTPUT / 'prior_strength.csv'}")

    # The point of the exercise: promoted and fitted priors must sit on
    # different timescales, or the split the peer predicted isn't real.
    assert prom["matchdays_to_10pt"].median() < 8, "promoted prior not noticeably weak"
    if len(fit_reached):
        assert fit_reached["matchdays_to_10pt"].median() > prom["matchdays_to_10pt"].median(), \
            "fitted priors should be slower to move than promoted ones"
    print("OK  promoted priors move faster than fitted ones")


if __name__ == "__main__":
    main()
