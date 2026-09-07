"""
Dixon-Coles match model.

Each team gets an attack and a defence strength; one global home advantage
applies to every ground. The rho term corrects the low-scoring corner of the
scoreline grid (0-0, 1-0, 0-1, 1-1), which independent Poissons get wrong.

Two fits are run and blended: one on actual goals (with the rho correction,
which needs integer scores) and one on xG (continuous, more predictive of
future results). Blending strength parameters keeps both fits individually
valid -- a single hybrid target would break the likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

import config


# ------------------------------------------------------------------ helpers


def time_weights(dates: pd.Series, reference: pd.Timestamp, xi: float) -> np.ndarray:
    """exp(-xi * days_ago). Recent matches dominate; old ones fade smoothly."""
    days = (reference - pd.to_datetime(dates)).dt.total_seconds().to_numpy() / 86400.0
    days = np.clip(days, 0.0, None)
    return np.exp(-xi * days)


def _poisson_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Poisson log-density, valid for continuous k so xG can be fitted directly."""
    lam = np.clip(lam, 1e-10, None)
    return k * np.log(lam) - lam - gammaln(k + 1.0)


def _tau(hg: np.ndarray, ag: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction. Identity everywhere except the 2x2 corner."""
    out = np.ones_like(lh, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    out[m00] = 1.0 - lh[m00] * la[m00] * rho
    out[m01] = 1.0 + lh[m01] * rho
    out[m10] = 1.0 + la[m10] * rho
    out[m11] = 1.0 - rho
    return np.clip(out, 1e-10, None)


# ------------------------------------------------------------------ model


@dataclass
class DCParams:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    n_matches: int = 0
    log_likelihood: float = float("nan")
    source: str = "goals"
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.index:
            self.index = {t: i for i, t in enumerate(self.teams)}

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        h, a = self.index[home], self.index[away]
        lam_h = np.exp(self.attack[h] - self.defence[a] + self.home_adv)
        lam_a = np.exp(self.attack[a] - self.defence[h])
        return float(lam_h), float(lam_a)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"team": self.teams, "attack": self.attack, "defence": self.defence}
        ).sort_values("attack", ascending=False, ignore_index=True)


def _unpack(params: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Attack strengths sum to zero, otherwise the model is unidentifiable."""
    attack = np.empty(n)
    attack[: n - 1] = params[: n - 1]
    attack[n - 1] = -attack[: n - 1].sum()
    defence = params[n - 1 : 2 * n - 1]
    return attack, defence, float(params[-2]), float(params[-1])


def _neg_log_likelihood(
    params: np.ndarray,
    n: int,
    hi: np.ndarray,
    ai: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    w: np.ndarray,
    use_tau: bool,
) -> float:
    attack, defence, home_adv, rho = _unpack(params, n)

    lam_h = np.exp(attack[hi] - defence[ai] + home_adv)
    lam_a = np.exp(attack[ai] - defence[hi])

    ll = _poisson_logpmf(hg, lam_h) + _poisson_logpmf(ag, lam_a)
    if use_tau:
        ll = ll + np.log(_tau(hg, ag, lam_h, lam_a, rho))

    total = float(np.sum(w * ll))
    return -total if np.isfinite(total) else 1e10


def fit(
    matches: pd.DataFrame,
    *,
    target: str = "goals",
    xi: float | None = None,
    reference_date: pd.Timestamp | None = None,
    teams: list[str] | None = None,
) -> DCParams:
    """
    Fit attack/defence/home-advantage/rho by weighted maximum likelihood.

    matches needs: date, home_team, away_team, and either
    home_goals/away_goals (target="goals") or home_xg/away_xg (target="xg").
    """
    xi = config.XI if xi is None else xi
    df = matches.dropna(subset=["home_team", "away_team"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    if target == "goals":
        hcol, acol, use_tau = "home_goals", "away_goals", True
    elif target == "xg":
        # rho describes integer-scoreline dependence; it has no meaning for xG.
        hcol, acol, use_tau = "home_xg", "away_xg", False
    else:
        raise ValueError(f"target must be 'goals' or 'xg', got {target!r}")

    df = df.dropna(subset=[hcol, acol])
    if df.empty:
        raise ValueError(f"no rows with both {hcol} and {acol}")

    if teams is None:
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    known = df["home_team"].isin(idx) & df["away_team"].isin(idx)
    df = df[known]

    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    hg = df[hcol].to_numpy(dtype=float)
    ag = df[acol].to_numpy(dtype=float)

    reference = pd.to_datetime(reference_date) if reference_date is not None else df["date"].max()
    w = time_weights(df["date"], reference, xi)

    x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25], [config.RHO_INIT]])
    bounds = [(-3.0, 3.0)] * (n - 1) + [(-3.0, 3.0)] * n + [(-1.0, 1.0), (-0.3, 0.3)]

    res = minimize(
        _neg_log_likelihood,
        x0,
        args=(n, hi, ai, hg, ag, w, use_tau),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    attack, defence, home_adv, rho = _unpack(res.x, n)
    return DCParams(
        teams=list(teams),
        attack=attack,
        defence=defence,
        home_adv=home_adv,
        rho=rho if use_tau else 0.0,
        n_matches=len(df),
        log_likelihood=-res.fun,
        source=target,
    )


def blend(goals_fit: DCParams, xg_fit: DCParams, w_xg: float | None = None) -> DCParams:
    """
    Combine the two fits. xG carries more weight because it predicts future
    results better than the scorelines that happened to occur.
    """
    w_xg = config.XG_BLEND if w_xg is None else w_xg
    if goals_fit.teams != xg_fit.teams:
        raise ValueError("fits must cover the same teams in the same order")

    return DCParams(
        teams=goals_fit.teams,
        attack=(1 - w_xg) * goals_fit.attack + w_xg * xg_fit.attack,
        defence=(1 - w_xg) * goals_fit.defence + w_xg * xg_fit.defence,
        home_adv=(1 - w_xg) * goals_fit.home_adv + w_xg * xg_fit.home_adv,
        rho=goals_fit.rho,  # only the goals fit can estimate this
        n_matches=goals_fit.n_matches,
        source=f"blend(goals={1 - w_xg:.2f}, xg={w_xg:.2f})",
    )


def fit_blended(matches: pd.DataFrame, **kwargs) -> DCParams:
    """Fit on goals and on xG, then blend. Falls back to goals if xG is absent."""
    teams = kwargs.pop("teams", None)
    if teams is None:
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))

    goals_fit = fit(matches, target="goals", teams=teams, **kwargs)

    has_xg = {"home_xg", "away_xg"} <= set(matches.columns)
    if not has_xg or matches[["home_xg", "away_xg"]].dropna().empty:
        print("  ! no xG available, using goals only")
        return goals_fit

    xg_fit = fit(matches, target="xg", teams=teams, **kwargs)
    return blend(goals_fit, xg_fit)


# ------------------------------------------------------------------ prediction


def score_matrix(lam_h: float, lam_a: float, rho: float, max_goals: int | None = None) -> np.ndarray:
    """Joint probability of every scoreline up to max_goals, rho-corrected."""
    from scipy.stats import poisson

    mg = config.MAX_GOALS if max_goals is None else max_goals
    grid = np.arange(mg + 1)
    m = np.outer(poisson.pmf(grid, lam_h), poisson.pmf(grid, lam_a))

    m[0, 0] *= 1.0 - lam_h * lam_a * rho
    m[0, 1] *= 1.0 + lam_h * rho
    m[1, 0] *= 1.0 + lam_a * rho
    m[1, 1] *= 1.0 - rho

    m = np.clip(m, 0.0, None)
    return m / m.sum()


def outcome_probs(params: DCParams, home: str, away: str) -> tuple[float, float, float]:
    """(home win, draw, away win)."""
    lam_h, lam_a = params.lambdas(home, away)
    m = score_matrix(lam_h, lam_a, params.rho)
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


# ------------------------------------------------------------------ self-check

if __name__ == "__main__":
    # Generate matches from known strengths, refit, and check they come back.
    rng = np.random.default_rng(7)
    teams = [f"T{i:02d}" for i in range(20)]
    n = len(teams)

    true_attack = rng.normal(0, 0.35, n)
    true_attack -= true_attack.mean()
    true_defence = rng.normal(0, 0.30, n)
    true_home = 0.26

    rows = []
    start = pd.Timestamp("2024-08-01")
    for rep in range(6):  # 6 round-robins keeps the estimates stable
        for h in range(n):
            for a in range(n):
                if h == a:
                    continue
                lh = np.exp(true_attack[h] - true_defence[a] + true_home)
                la = np.exp(true_attack[a] - true_defence[h])
                rows.append(
                    {
                        "date": start + pd.Timedelta(days=rep * 60 + (h + a) % 60),
                        "home_team": teams[h],
                        "away_team": teams[a],
                        "home_goals": rng.poisson(lh),
                        "away_goals": rng.poisson(la),
                    }
                )
    df = pd.DataFrame(rows)

    # xi=0 so every synthetic match counts equally against the known truth.
    p = fit(df, target="goals", xi=0.0, teams=teams)

    atk_corr = np.corrcoef(p.attack, true_attack)[0, 1]
    def_corr = np.corrcoef(p.defence, true_defence)[0, 1]
    assert atk_corr > 0.95, f"attack recovery poor: r={atk_corr:.3f}"
    assert def_corr > 0.95, f"defence recovery poor: r={def_corr:.3f}"
    assert abs(p.home_adv - true_home) < 0.06, f"home_adv off: {p.home_adv:.3f}"
    assert abs(p.attack.sum()) < 1e-8, "attack constraint violated"

    m = score_matrix(1.5, 1.1, p.rho)
    assert abs(m.sum() - 1.0) < 1e-9, "score matrix does not sum to 1"
    hw, d, aw = outcome_probs(p, teams[0], teams[1])
    assert abs(hw + d + aw - 1.0) < 1e-9, "outcome probs do not sum to 1"

    # Time decay must actually down-weight old matches. The half-life is
    # derived from config.XI rather than hard-coded, so a backtest that
    # retunes the decay does not leave a stale number asserted here.
    half_life = np.log(2) / config.XI
    ref = start + pd.Timedelta(days=half_life)
    w = time_weights(pd.Series([start, ref]), ref, config.XI)
    assert abs(w[1] - 1.0) < 1e-9, f"today's match must carry full weight: {w[1]}"
    assert abs(w[0] - 0.5) < 1e-6, f"a match one half-life old must weigh half: {w[0]}"
    # Sanity on the tuned value itself: a season-long memory, not a decade.
    assert 30 < half_life < 800, f"implausible half-life of {half_life:.0f} days from XI={config.XI}"

    print(f"OK  fitted {p.n_matches} matches | attack r={atk_corr:.3f} "
          f"defence r={def_corr:.3f} home_adv={p.home_adv:.3f} rho={p.rho:.3f}")
