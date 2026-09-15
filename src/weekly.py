"""
The weekly-updating forecast, logged beside the pre-season one.

    python -m src.weekly

Runs the same model twice: once ignoring this season's results (the pre-season
forecast, the one being graded in May) and once conditioned on every match
played so far (folded into the Dixon-Coles fit, promoted clubs blended toward
their real games). Appends both to output/series_log.csv, one row per club per
run, so in May the two series can be compared -- the test being whether not
updating cost anything.

Safe to run every week. Rows are keyed by run date and series name.
"""

from __future__ import annotations

import pandas as pd

import config
from src import run_pipeline as rp
from src.live_results import fetch_played

_LOG = config.OUTPUT / "series_log.csv"
_COLS = ["run_date", "series", "matchday", "team", "exp_points", "exp_rank",
         "P_title", "P_top4", "P_bottom3"]


def _rows(table: pd.DataFrame, series: str, run_date: str, matchday: int) -> pd.DataFrame:
    out = table[["team", "exp_points", "exp_rank", "P_title", "P_top4", "P_bottom3"]].copy()
    out.insert(0, "matchday", matchday)
    out.insert(0, "series", series)
    out.insert(0, "run_date", run_date)
    return out[_COLS]


def _append(rows: pd.DataFrame) -> None:
    if _LOG.exists():
        prior = pd.read_csv(_LOG)
        # A re-run on the same day replaces that day's rows rather than
        # doubling them. `.isin` on an Index already returns an ndarray.
        new_keys = set(map(tuple, rows[["run_date", "series"]].to_numpy()))
        keep = [tuple(k) not in new_keys
                for k in prior[["run_date", "series"]].to_numpy()]
        rows = pd.concat([prior[keep], rows], ignore_index=True)
    rows.to_csv(_LOG, index=False)


def main() -> None:
    run_date = config.TODAY
    played = fetch_played()
    matchday = int(played["matchweek"].max()) if len(played) else 0
    print(f"{len(played)} matches played, through matchweek {matchday}\n")

    print("=" * 70)
    pre, _, _ = rp.main(write=True)                    # pre-season keeps the canonical output/
    print("=" * 70)
    wk, _, _ = rp.main(played=played, write=False)     # weekly is logged only
    print("=" * 70)

    _append(pd.concat([
        _rows(pre, "preseason", run_date, matchday),
        _rows(wk, "weekly", run_date, matchday),
    ], ignore_index=True))

    merged = pre[["team", "exp_points", "P_bottom3"]].merge(
        wk[["team", "exp_points", "P_bottom3"]], on="team", suffixes=("_pre", "_wk"))
    merged["move"] = merged["exp_points_wk"] - merged["exp_points_pre"]
    merged["releg_move"] = merged["P_bottom3_wk"] - merged["P_bottom3_pre"]
    merged = merged.reindex(merged["move"].abs().sort_values(ascending=False).index)

    print("\nbiggest gap between the two series (weekly minus pre-season):")
    for r in merged.head(8).itertuples(index=False):
        print(f"  {r.team:18s} {r.move:+5.1f} pts   relegation {r.releg_move:+.1%}")
    print(f"\nlogged to {_LOG}  ({len(pd.read_csv(_LOG))} rows total)")

    if matchday >= 1:
        # The conditioning has to actually condition, and in the direction the
        # table says: a club that started well should look safer in the weekly
        # series, one that started badly more exposed. If both series agree the
        # played matches are being ignored somewhere.
        assert merged["move"].abs().max() > 0.5, "weekly series identical to pre-season"
        pts_so_far = {t: 0 for t in config.TEAMS}
        for r in played.itertuples(index=False):
            if r.home_goals > r.away_goals:
                pts_so_far[r.home_team] += 3
            elif r.home_goals < r.away_goals:
                pts_so_far[r.away_team] += 3
            else:
                pts_so_far[r.home_team] += 1
                pts_so_far[r.away_team] += 1
        by_pts = sorted(config.TEAMS, key=lambda t: pts_so_far[t])
        worst, best = by_pts[0], by_pts[-1]
        d = merged.set_index("team")
        assert d.loc[best, "releg_move"] <= 0.02, f"{best} started top but weekly made them worse"
        assert d.loc[worst, "releg_move"] >= -0.02, f"{worst} started bottom but weekly made them safer"
        print("OK  conditioning moves clubs in the direction of their results")


if __name__ == "__main__":
    main()
