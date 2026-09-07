"""
Monte Carlo season simulation.

Two things separate this from naive repeated sampling:

1. Team strength is redrawn once per simulated season, not per match. A side
   sampled strong stays strong all year -- which is what makes a title race or
   a collapse possible at all.
2. Strength then drifts as a random walk through the matchweeks, so form
   persists. Treating matches as independent produces rank distributions that
   are far too narrow: nobody ever goes on a ten-game run, good or bad.

Uncertainty is per-team. A club with a settled manager and five seasons of
Premier League data gets a narrow band; a promoted side with a new manager
and a non-UEFA prior gets a wide one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.dixon_coles import DCParams


def team_sigma(teams: list[str], flags: dict[str, dict] | None = None) -> np.ndarray:
    """
    Per-team strength uncertainty. Reasons compound: a promoted club with a
    new manager is less knowable than either on its own.

    A thin squad widens the band too, and for a different reason than the
    others: not ignorance about the club, but genuine fragility. Lose a key
    player for two months and there is no equivalent replacement, so the
    season can fall apart -- or the same squad can stay fit and finish where
    its quality says. That is a spread, not a lower average, which is why it
    belongs here and not in the fatigue multiplier. The fatigue channel
    already handles the average during congested weeks; this handles the luck
    of who gets injured, which does not wait for a busy fixture list.
    """
    flags = flags or {}
    out = np.full(len(teams), config.SIGMA_BASE, dtype=float)

    for i, team in enumerate(teams):
        f = flags.get(team, {})
        extra = []
        if f.get("promoted"):
            extra.append(config.SIGMA_PROMOTED)
        if f.get("new_manager"):
            extra.append(config.SIGMA_NEW_MANAGER)
        if f.get("unproven_background"):
            extra.append(config.SIGMA_NON_UEFA_PRIOR)

        # How much the fit actually knows about this club. A side with one
        # Premier League season behind it is not as knowable as one with
        # three, however tidy that single season looked.
        evidence = float(f.get("evidence", 1.0))
        if evidence < 1.0:
            extra.append(config.SIGMA_THIN_EVIDENCE * (1.0 - evidence))

        # Continuous, unlike the flags: scaled by how thin the squad is
        # relative to the league, so an averagely-stocked club adds nothing.
        thinness = float(f.get("thinness", 0.0))
        if thinness > 0:
            extra.append(config.SIGMA_THIN_SQUAD * thinness)

        if extra:
            # Independent sources of ignorance add in quadrature.
            out[i] = float(np.sqrt(config.SIGMA_BASE**2 + sum(e**2 for e in extra)))
    return out


def _proven_in_europe(team: str, mgr: dict, priors: dict | None, as_of: pd.Timestamp) -> bool:
    """
    Has this manager already completed a full season in a strong European
    first division -- here, or anywhere else?

    Two ways to qualify, and either is enough:

      - a previous stint of at least a full season's matches in a league at or
        above PROVEN_LEAGUE_MIN_WEIGHT;
      - a full season already served at the current club, which for every club
        in this file means the Premier League itself. A manager appointed a
        year ago is not an unknown quantity just because their CV before that
        was unusual.
    """
    from src.features import league_weight

    for stint in mgr.get("prior_stints", []):
        lw = league_weight(stint["league"])
        if lw is None or lw < config.PROVEN_LEAGUE_MIN_WEIGHT:
            continue
        matches = (priors or {}).get(team, {}).get(stint["club"], {}).get("matches")
        if matches is None:
            # No match count on file: fall back to the stint's own span.
            matches = (float(stint.get("end", 0)) - float(stint.get("start", 0))) * 38
        if matches >= config.PROVEN_LEAGUE_MIN_MATCHES:
            return True

    since = mgr.get("since")
    if since:
        days_in_post = (as_of - pd.Timestamp(since)).days
        if days_in_post >= 300:  # a full Premier League campaign, start to finish
            return True

    return False


def default_flags(priors: dict | None = None, as_of: pd.Timestamp | None = None) -> dict[str, dict]:
    """
    Uncertainty flags derived from config: who is new, promoted, thinly
    evidenced, or arriving from a background this league cannot vouch for.

    priors is the manager-prior file, used only to read real match counts for
    the "full season" test; without it the test falls back to stint spans.
    """
    as_of = pd.Timestamp(config.TODAY) if as_of is None else as_of
    flags: dict[str, dict] = {}

    for team in config.TEAMS:
        mgr = config.MANAGERS.get(team, {})
        f = {
            "promoted": team in config.PROMOTED,
            "new_manager": mgr.get("status") in ("new", "interim_to_full"),
            "unproven_background": bool(mgr) and not _proven_in_europe(team, mgr, priors, as_of),
        }
        if any(f.values()):
            flags[team] = f
    return flags


def _apply_rho(hg, ag, lam_h, lam_a, rho: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """
    Dixon-Coles low-score correction, applied to sampled scorelines.

    The fit estimates rho because independent Poissons get the 2x2 corner of
    the scoreline grid wrong -- real football produces more 0-0 and 1-1 draws,
    and fewer 1-0 and 0-1 wins, than two independent goal counts predict.
    Sampling without it quietly throws that estimate away.

    Rejection sampling: only the four affected scorelines are touched, and
    only their relative probabilities change, so a draw is kept with
    probability tau/max(tau) and redrawn otherwise. Everything at 2+ goals
    passes through untouched, which is the vast majority of matches.
    """
    if abs(rho) < 1e-9:
        return hg, ag

    # Highest tau any outcome of THIS match can reach, so acceptance is scaled
    # per match rather than by a league-wide maximum -- a global ceiling would
    # drag every high-scoring match into needless resampling.
    ceiling = np.maximum.reduce([
        np.ones_like(lam_h),
        1.0 - lam_h * lam_a * rho,
        1.0 + lam_h * rho,
        1.0 + lam_a * rho,
        np.full_like(lam_h, 1.0 - rho),
    ])

    # Each draw is tested exactly once. Re-testing an already-accepted draw on
    # a later pass would quietly rebuild the very bias this is here to remove:
    # a 0-0 sits at the ceiling and always survives, while a 2-0 with tau=1
    # keeps facing the same coin and gets worn away, so low scores pile up far
    # beyond what rho asks for.
    pending = np.ones(hg.shape, dtype=bool)

    for _ in range(24):  # convergence is geometric; the cap guards a pathological rho
        idx = np.flatnonzero(pending)
        if idx.size == 0:
            break

        h, a = hg[idx], ag[idx]
        lh, la = lam_h[idx], lam_a[idx]

        tau = np.ones(idx.size, dtype=float)
        tau[(h == 0) & (a == 0)] = (1.0 - lh * la * rho)[(h == 0) & (a == 0)]
        tau[(h == 0) & (a == 1)] = (1.0 + lh * rho)[(h == 0) & (a == 1)]
        tau[(h == 1) & (a == 0)] = (1.0 + la * rho)[(h == 1) & (a == 0)]
        tau[(h == 1) & (a == 1)] = 1.0 - rho
        tau = np.clip(tau, 0.0, None)

        accept = rng.random(idx.size) <= (tau / ceiling[idx])
        pending[idx[accept]] = False

        redraw = idx[~accept]
        if redraw.size:
            hg[redraw] = rng.poisson(np.clip(lam_h[redraw], 1e-6, 15.0))
            ag[redraw] = rng.poisson(np.clip(lam_a[redraw], 1e-6, 15.0))

    return hg, ag


def simulate(
    params: DCParams,
    fixtures: pd.DataFrame,
    *,
    n_sims: int | None = None,
    sigma: np.ndarray | None = None,
    flags: dict[str, dict] | None = None,
    seed: int | None = None,
    played: pd.DataFrame | None = None,
) -> dict:
    """
    Simulate the season n_sims times.

    fixtures needs: home_team, away_team, matchweek, and optionally
    home_fatigue / away_fatigue multipliers from features.build_fatigue().

    played optionally carries already-completed matches (home_goals /
    away_goals), so the same code predicts a season in progress. For a
    pre-season run it is simply empty.
    """
    n_sims = config.N_SIMULATIONS if n_sims is None else n_sims
    seed = config.RANDOM_SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    teams = params.teams
    idx = params.index
    n_teams = len(teams)

    if sigma is None:
        sigma = team_sigma(teams, flags if flags is not None else default_flags())

    # --- season-long strength draw ---------------------------------------
    atk = params.attack[None, :] + rng.normal(0.0, 1.0, (n_sims, n_teams)) * sigma[None, :]
    dfn = params.defence[None, :] + rng.normal(0.0, 1.0, (n_sims, n_teams)) * sigma[None, :]

    # --- form as a random walk over matchweeks ---------------------------
    mw = fixtures["matchweek"].to_numpy(dtype=int)
    n_mw = int(mw.max()) + 1
    form = np.cumsum(
        rng.normal(0.0, config.FORM_DRIFT_SIGMA, (n_sims, n_mw, n_teams)), axis=1
    )

    points = np.zeros((n_sims, n_teams))
    gf = np.zeros((n_sims, n_teams))
    ga = np.zeros((n_sims, n_teams))

    # --- carry in results that already happened --------------------------
    if played is not None and len(played):
        for row in played.itertuples(index=False):
            h, a = idx.get(row.home_team), idx.get(row.away_team)
            if h is None or a is None:
                continue
            hg, ag = int(row.home_goals), int(row.away_goals)
            gf[:, h] += hg
            ga[:, h] += ag
            gf[:, a] += ag
            ga[:, a] += hg
            if hg > ag:
                points[:, h] += config.POINTS_WIN
                points[:, a] += config.POINTS_LOSS
            elif hg < ag:
                points[:, a] += config.POINTS_WIN
                points[:, h] += config.POINTS_LOSS
            else:
                points[:, h] += config.POINTS_DRAW
                points[:, a] += config.POINTS_DRAW

    # --- simulate the remaining fixtures ---------------------------------
    # Sorted by matchweek so the per-week snapshots below are cumulative in
    # the right order. Stable, so date order within a week is untouched, and
    # totals are unaffected either way -- addition does not care. Sorting
    # explicitly rather than trusting the caller keeps this correct if a
    # postponed match ever lands out of date order.
    order = np.argsort(mw, kind="stable")
    fixtures = fixtures.iloc[order].reset_index(drop=True)
    mw = mw[order]

    hi = fixtures["home_team"].map(idx).to_numpy()
    ai = fixtures["away_team"].map(idx).to_numpy()
    h_fat = fixtures.get("home_fatigue", pd.Series(1.0, index=fixtures.index)).to_numpy(dtype=float)
    a_fat = fixtures.get("away_fatigue", pd.Series(1.0, index=fixtures.index)).to_numpy(dtype=float)

    if np.isnan(hi.astype(float)).any() or np.isnan(ai.astype(float)).any():
        missing = set(fixtures["home_team"]) | set(fixtures["away_team"])
        raise KeyError(f"fixtures reference unknown teams: {missing - set(teams)}")
    hi = hi.astype(int)
    ai = ai.astype(int)

    # Week-by-week snapshots of the table as it stands. The final standings
    # alone cannot say whether a club led in October and faded, or climbed all
    # season -- and that is the shape a season actually has. Summarised at each
    # week rather than keeping every draw, because the full cube would be
    # n_sims x 38 x 20 and is not needed to plot a trend with an uncertainty band.
    weeks = sorted(set(int(w) for w in mw))
    track = {
        "matchweeks": weeks,
        "points_mean": np.zeros((len(weeks), n_teams)),
        "points_p10": np.zeros((len(weeks), n_teams)),
        "points_p90": np.zeros((len(weeks), n_teams)),
        "rank_mean": np.zeros((len(weeks), n_teams)),
    }
    week_slot = {w: i for i, w in enumerate(weeks)}

    sims = np.arange(n_sims)
    for k in range(len(fixtures)):
        h, a, w = hi[k], ai[k], mw[k]
        fh = form[:, w, h]
        fa = form[:, w, a]

        # Good form lifts both ends: sharper going forward, tighter at the back.
        lam_h = np.exp(atk[:, h] + fh - (dfn[:, a] + fa) + params.home_adv) * h_fat[k]
        lam_a = np.exp(atk[:, a] + fa - (dfn[:, h] + fh)) * a_fat[k]

        hg = rng.poisson(np.clip(lam_h, 1e-6, 15.0))
        ag = rng.poisson(np.clip(lam_a, 1e-6, 15.0))
        hg, ag = _apply_rho(hg, ag, lam_h, lam_a, params.rho, rng)

        gf[:, h] += hg
        ga[:, h] += ag
        gf[:, a] += ag
        ga[:, a] += hg

        home_win = hg > ag
        away_win = ag > hg
        draw = ~(home_win | away_win)

        points[sims[home_win], h] += config.POINTS_WIN
        points[sims[away_win], a] += config.POINTS_WIN
        points[sims[draw], h] += config.POINTS_DRAW
        points[sims[draw], a] += config.POINTS_DRAW

        # End of a matchweek: record where the table stands.
        if k + 1 == len(fixtures) or mw[k + 1] != w:
            s = week_slot[int(w)]
            track["points_mean"][s] = points.mean(axis=0)
            track["points_p10"][s] = np.percentile(points, 10, axis=0)
            track["points_p90"][s] = np.percentile(points, 90, axis=0)
            track["rank_mean"][s] = _rank(points, gf - ga, gf, rng).mean(axis=0)

    ranks = _rank(points, gf - ga, gf, rng)
    return {
        "teams": teams,
        "points": points,
        "goal_diff": gf - ga,
        "goals_for": gf,
        "ranks": ranks,
        "n_sims": n_sims,
        "sigma": sigma,
        "track": track,
    }


def _rank(points, goal_diff, goals_for, rng) -> np.ndarray:
    """Premier League order: points, then goal difference, then goals scored."""
    # Composite sort key; the noise term breaks exact ties at random rather
    # than by alphabetical accident.
    key = (
        points * 1e7
        + (goal_diff + 200.0) * 1e3
        + goals_for
        + rng.random(points.shape) * 1e-3
    )
    order = np.argsort(-key, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, points.shape[1] + 1)[None, :], axis=1)
    return ranks


def summarise(result: dict) -> pd.DataFrame:
    """Per-team rank distribution: the actual deliverable."""
    teams = result["teams"]
    ranks = result["ranks"]
    points = result["points"]

    rows = []
    for i, team in enumerate(teams):
        r = ranks[:, i]
        p = points[:, i]
        rows.append(
            {
                "team": team,
                "exp_points": p.mean(),
                "points_p10": np.percentile(p, 10),
                "points_p90": np.percentile(p, 90),
                "exp_rank": r.mean(),
                "median_rank": np.median(r),
                "rank_p10": np.percentile(r, 10),
                "rank_p90": np.percentile(r, 90),
                "P_title": (r == 1).mean(),
                "P_top4": (r <= 4).mean(),
                "P_top6": (r <= 6).mean(),
                "P_bottom3": (r >= 18).mean(),
                "sigma": result["sigma"][i],
            }
        )
    return pd.DataFrame(rows).sort_values("exp_rank", ignore_index=True)


def rank_matrix(result: dict) -> pd.DataFrame:
    """Full P(team finishes in position k) grid."""
    teams = result["teams"]
    ranks = result["ranks"]
    n = len(teams)
    grid = np.zeros((n, n))
    for i in range(n):
        counts = np.bincount(ranks[:, i], minlength=n + 1)[1:]
        grid[i] = counts / result["n_sims"]
    df = pd.DataFrame(grid, index=teams, columns=range(1, n + 1))
    return df.loc[df.mul(df.columns).sum(axis=1).sort_values().index]


# ------------------------------------------------------------------ self-check

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    teams = [f"T{i:02d}" for i in range(20)]
    attack = np.linspace(0.45, -0.45, 20)
    defence = np.linspace(0.40, -0.40, 20)
    p = DCParams(teams=teams, attack=attack, defence=defence, home_adv=0.26, rho=-0.04)

    fixtures = pd.DataFrame(
        [
            {"home_team": teams[h], "away_team": teams[a], "matchweek": (h + a) % 38}
            for h in range(20)
            for a in range(20)
            if h != a
        ]
    )

    res = simulate(p, fixtures, n_sims=2000, seed=42, flags={})
    tbl = summarise(res)

    assert len(fixtures) == 380, f"expected 380 fixtures, got {len(fixtures)}"
    assert res["ranks"].min() == 1 and res["ranks"].max() == 20, "rank range wrong"
    # Every simulated season must be a permutation of 1..20.
    assert (np.sort(res["ranks"], axis=1) == np.arange(1, 21)).all(), "ranks not a permutation"
    # Total points must equal 3*decisive + 2*draws, bounded by 2280 and 3420.
    tot = res["points"].sum(axis=1)
    assert (tot >= 380 * 2).all() and (tot <= 380 * 3).all(), "points total out of range"
    assert abs(tbl["P_title"].sum() - 1.0) < 1e-9, "title probabilities must sum to 1"
    assert abs(tbl["P_top4"].sum() - 4.0) < 1e-9, "top-4 probabilities must sum to 4"
    # Strongest side must beat the weakest on expected points.
    assert tbl.iloc[0]["exp_points"] > tbl.iloc[-1]["exp_points"] + 20, "no signal"

    # Squad thinness widens the band, continuously and only below average.
    sig = team_sigma(["Deep", "Average", "Thin"], {
        "Thin": {"thinness": 1.0},
        "Average": {"thinness": 0.0},
    })
    assert sig[0] == config.SIGMA_BASE, "an unflagged club keeps the base band"
    assert sig[1] == config.SIGMA_BASE, "thinness 0 must add nothing at all"
    assert sig[2] > sig[1], f"a thin squad must be less predictable: {sig[2]:.3f} vs {sig[1]:.3f}"
    # Half as thin, less than half the extra width -- quadrature, not linear.
    half = team_sigma(["X"], {"X": {"thinness": 0.5}})[0]
    assert sig[1] < half < sig[2], f"thinness must scale continuously: {half:.3f}"

    # It stacks with the flag-based reasons rather than replacing them.
    both = team_sigma(["X"], {"X": {"promoted": True, "thinness": 1.0}})[0]
    promoted_only = team_sigma(["X"], {"X": {"promoted": True}})[0]
    assert both > promoted_only, "thinness must compound with the other reasons"

    # Week-by-week tracking must actually describe a season unfolding.
    tr = res["track"]
    pm = tr["points_mean"]
    n_weeks = fixtures["matchweek"].nunique()
    assert pm.shape == (n_weeks, 20), f"expected {n_weeks} weeks x 20 clubs, got {pm.shape}"
    assert tr["matchweeks"] == sorted(fixtures["matchweek"].unique()), "weeks out of order"
    # Points only ever accumulate, so every club's line must rise monotonically.
    assert (np.diff(pm, axis=0) >= -1e-9).all(), "cumulative points went backwards"
    # A matchweek is 10 fixtures, so the league gains 20-30 points a week; after
    # 38 weeks the totals must land on the same place summarise() reports.
    assert abs(pm[-1].sum() - res["points"].mean(axis=0).sum()) < 1e-6, \
        "final tracked week disagrees with the finished season"
    assert 0 < pm[0].sum() <= 30, f"opening week total implausible: {pm[0].sum():.1f}"
    # The band has to bracket the mean, and widen as more football is played.
    assert (tr["points_p10"] <= pm + 1e-9).all() and (tr["points_p90"] >= pm - 1e-9).all(), \
        "percentile band does not contain the mean"
    early = (tr["points_p90"][0] - tr["points_p10"][0]).mean()
    late = (tr["points_p90"][-1] - tr["points_p10"][-1]).mean()
    assert late > early, f"uncertainty should grow over a season: {early:.1f} -> {late:.1f}"
    assert (tr["rank_mean"] >= 1).all() and (tr["rank_mean"] <= 20).all(), "rank out of range"

    # Thin match evidence widens the band: a club the fit barely saw is not
    # as knowable as one it saw every week.
    ev = team_sigma(["Full", "Half"], {"Half": {"evidence": 0.5}})
    assert ev[0] == config.SIGMA_BASE, "a club with full evidence keeps the base band"
    assert ev[1] > ev[0], f"one season of data must be less certain: {ev[1]:.3f} vs {ev[0]:.3f}"

    # The rho correction must actually bend the low-score corner: more 0-0 and
    # 1-1, fewer 1-0 and 0-1, exactly as Dixon-Coles says.
    r = np.random.default_rng(3)
    lam = np.full(200_000, 1.3)
    raw_h, raw_a = r.poisson(lam), r.poisson(lam)
    cor_h, cor_a = _apply_rho(raw_h.copy(), raw_a.copy(), lam, lam, -0.12, r)
    draws_raw = ((raw_h == 0) & (raw_a == 0)).mean() + ((raw_h == 1) & (raw_a == 1)).mean()
    draws_cor = ((cor_h == 0) & (cor_a == 0)).mean() + ((cor_h == 1) & (cor_a == 1)).mean()
    edge_raw = ((raw_h == 1) & (raw_a == 0)).mean() + ((raw_h == 0) & (raw_a == 1)).mean()
    edge_cor = ((cor_h == 1) & (cor_a == 0)).mean() + ((cor_h == 0) & (cor_a == 1)).mean()
    assert draws_cor > draws_raw, f"negative rho must add low draws: {draws_cor:.4f} vs {draws_raw:.4f}"
    assert edge_cor < edge_raw, f"negative rho must remove 1-0/0-1: {edge_cor:.4f} vs {edge_raw:.4f}"
    # The tau correction is built so the distribution still sums to one, which
    # means every scoreline outside the 2x2 corner keeps its plain Poisson
    # probability. An earlier version of the sampler re-tested draws it had
    # already accepted, which piled up low scores and dragged 2-0 down by a
    # fifth -- this is the check that caught it.
    from scipy.stats import poisson as _poisson
    tau_map = {(0, 0): 1 - 1.3 * 1.3 * -0.12, (0, 1): 1 + 1.3 * -0.12,
               (1, 0): 1 + 1.3 * -0.12, (1, 1): 1 - -0.12}
    for score in [(0, 0), (1, 1), (1, 0), (2, 0), (2, 1), (3, 2)]:
        want = _poisson.pmf(score[0], 1.3) * _poisson.pmf(score[1], 1.3) * tau_map.get(score, 1.0)
        got = ((cor_h == score[0]) & (cor_a == score[1])).mean()
        assert abs(got - want) < 0.004, f"{score}: sampler {got:.4f} vs Dixon-Coles {want:.4f}"
    # rho = 0 is a no-op, not a resample.
    same_h, same_a = _apply_rho(raw_h.copy(), raw_a.copy(), lam, lam, 0.0, r)
    assert (same_h == raw_h).all() and (same_a == raw_a).all(), "rho=0 must change nothing"

    # Form drift must widen the spread versus independent matches.
    saved = config.FORM_DRIFT_SIGMA
    config.FORM_DRIFT_SIGMA = 0.0
    flat = simulate(p, fixtures, n_sims=2000, seed=42, flags={})
    config.FORM_DRIFT_SIGMA = saved
    spread_drift = res["points"].std(axis=0).mean()
    spread_flat = flat["points"].std(axis=0).mean()
    assert spread_drift > spread_flat, (
        f"form drift should widen spread: {spread_drift:.2f} vs {spread_flat:.2f}"
    )

    print(f"OK  {res['n_sims']} sims | champion {tbl.iloc[0]['team']} "
          f"{tbl.iloc[0]['exp_points']:.1f} pts P(title)={tbl.iloc[0]['P_title']:.1%} | "
          f"points sd {spread_flat:.2f} -> {spread_drift:.2f} with form drift")
