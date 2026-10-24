"""Dixon-Coles model: team ratings in, scoreline probabilities out."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MATCHES_PATH = Path("data/processed/matches.parquet")
XI = 0.0019  # time decay per day: a match from a year ago counts about half as much
MAX_GOALS = 10  # the scoreline grid runs from 0-0 up to 10-10


@dataclass
class FittedModel:
    teams: list[str]
    attack: np.ndarray  # higher = scores more
    defence: np.ndarray  # higher = concedes more
    intercept: float
    home_adv: float
    champ_effect: float  # goal-rate shift for Championship matches
    rho: float  # Dixon-Coles low-score correction

    def expected_goals(self, home: str, away: str, league: str = "EPL") -> tuple[float, float]:
        h, a = self.teams.index(home), self.teams.index(away)
        shift = self.champ_effect if league == "Championship" else 0.0
        lam = np.exp(self.intercept + self.home_adv + self.attack[h] + self.defence[a] + shift)
        mu = np.exp(self.intercept + self.attack[a] + self.defence[h] + shift)
        return float(lam), float(mu)

    def score_grid(self, home: str, away: str, league: str = "EPL") -> np.ndarray:
        """grid[i, j] = probability the match finishes home i - away j."""
        lam, mu = self.expected_goals(home, away, league)
        goals = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        grid[0, 0] *= 1 - lam * mu * self.rho
        grid[0, 1] *= 1 + lam * self.rho
        grid[1, 0] *= 1 + mu * self.rho
        grid[1, 1] *= 1 - self.rho
        return grid / grid.sum()


def fit(matches: pd.DataFrame, as_of: pd.Timestamp, xi: float = XI) -> FittedModel:
    """Fit on every match played before `as_of`, weighting recent matches more."""
    df = matches[matches["date"] < as_of]
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    h = df["home_team"].map(index).to_numpy()
    a = df["away_team"].map(index).to_numpy()
    x = df["home_goals"].to_numpy()
    y = df["away_goals"].to_numpy()
    champ = (df["league"] == "Championship").to_numpy(dtype=float)
    days_ago = (as_of - df["date"]).dt.days.to_numpy()
    w = np.exp(-xi * days_ago)

    is00, is01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
    is10, is11 = (x == 1) & (y == 0), (x == 1) & (y == 1)

    def unpack(p):
        att = p[:n] - p[:n].mean()  # centre ratings so they're identifiable
        dfn = p[n : 2 * n] - p[n : 2 * n].mean()
        return att, dfn, *p[2 * n :]

    def neg_log_likelihood(p):
        att, dfn, intercept, home_adv, champ_eff, rho = unpack(p)
        log_lam = intercept + home_adv + att[h] + dfn[a] + champ_eff * champ
        log_mu = intercept + att[a] + dfn[h] + champ_eff * champ
        lam, mu = np.exp(log_lam), np.exp(log_mu)

        tau = np.ones_like(lam)
        tau[is00] = 1 - lam[is00] * mu[is00] * rho
        tau[is01] = 1 + lam[is01] * rho
        tau[is10] = 1 + mu[is10] * rho
        tau[is11] = 1 - rho

        ll = w * (x * log_lam - lam + y * log_mu - mu + np.log(tau))

        # Gradient of the log-likelihood with respect to log_lam, log_mu and rho
        g_lam = x - lam
        g_mu = y - mu
        g_rho = np.zeros_like(lam)
        t = tau
        g_lam[is00] -= lam[is00] * mu[is00] * rho / t[is00]
        g_mu[is00] -= lam[is00] * mu[is00] * rho / t[is00]
        g_rho[is00] = -lam[is00] * mu[is00] / t[is00]
        g_lam[is01] += lam[is01] * rho / t[is01]
        g_rho[is01] = lam[is01] / t[is01]
        g_mu[is10] += mu[is10] * rho / t[is10]
        g_rho[is10] = mu[is10] / t[is10]
        g_rho[is11] = -1 / t[is11]
        g_lam, g_mu, g_rho = w * g_lam, w * g_mu, w * g_rho

        g_att = np.bincount(h, g_lam, n) + np.bincount(a, g_mu, n)
        g_def = np.bincount(a, g_lam, n) + np.bincount(h, g_mu, n)
        grad = np.concatenate([
            g_att - g_att.mean(),
            g_def - g_def.mean(),
            [g_lam.sum() + g_mu.sum(), g_lam.sum(), ((g_lam + g_mu) * champ).sum(), g_rho.sum()],
        ])
        return -ll.sum(), -grad

    start = np.concatenate([np.zeros(2 * n), [0.2, 0.25, 0.0, -0.05]])
    bounds = [(None, None)] * (2 * n + 3) + [(-0.2, 0.2)]
    result = minimize(neg_log_likelihood, start, jac=True, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        print(f"Warning: optimiser did not fully converge ({result.message})")

    att, dfn, intercept, home_adv, champ_eff, rho = unpack(result.x)
    return FittedModel(teams, att, dfn, intercept, home_adv, champ_eff, rho)


def current_teams(matches: pd.DataFrame, league: str) -> list[str]:
    season = matches["season"].max()
    rows = matches[(matches["season"] == season) & (matches["league"] == league)]
    return sorted(set(rows["home_team"]))


def print_ratings(model: FittedModel, teams: list[str]) -> None:
    idx = [model.teams.index(t) for t in teams]
    table = pd.DataFrame({
        "attack": model.attack[idx],
        "defence": -model.defence[idx],  # flipped so higher = better
    }, index=teams)
    table["overall"] = table["attack"] + table["defence"]
    print(table.sort_values("overall", ascending=False).round(2).to_string())


def print_match(model: FittedModel, home: str, away: str) -> None:
    for team in (home, away):
        if team not in model.teams:
            print(f"\nUnknown team '{team}'. Use the names exactly as in the table above.")
            return
    lam, mu = model.expected_goals(home, away)
    grid = model.score_grid(home, away)
    p_home, p_draw, p_away = np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()
    print(f"\n{home} v {away}")
    print(f"Expected goals: {lam:.2f} - {mu:.2f}")
    print(f"Home {p_home:.0%} | Draw {p_draw:.0%} | Away {p_away:.0%}")
    print("Most likely scores:")
    top = np.argsort(grid, axis=None)[::-1][:5]
    for i, j in zip(*np.unravel_index(top, grid.shape)):
        print(f"  {i}-{j}  {grid[i, j]:.1%}")


if __name__ == "__main__":
    matches = pd.read_parquet(MATCHES_PATH)
    as_of = matches["date"].max() + pd.Timedelta(days=1)
    model = fit(matches, as_of)
    print(f"Home advantage: {np.exp(model.home_adv):.2f}x goals | rho: {model.rho:.3f}\n")
    print_ratings(model, current_teams(matches, "EPL"))

    if len(sys.argv) == 3:  # e.g. uv run python -m score_predictor.model Arsenal Chelsea
        print_match(model, sys.argv[1], sys.argv[2])