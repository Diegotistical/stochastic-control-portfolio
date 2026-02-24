"""
Merton closed-form solution for optimal portfolio allocation.

Under full information, no transaction costs, and constant parameters,
the CRRA investor's optimal allocation is:

    π* = (μ − r) / ((1 − γ) σ²)

and the value function is:

    V(W, t) = (W^γ / γ) · exp(A · (T − t))

where A depends on γ, μ, σ, r.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class MertonSolution:
    """Container for Merton closed-form solution."""
    pi_star: NDArray[np.float64]       # (d,) optimal weights
    expected_utility: float           # E[U(W_T)]
    certainty_equivalent: float       # CE wealth
    value_function_coef: float        # coefficient A in exp(A*(T-t))


class MertonStrategy:
    """Merton (1969/1971) closed-form portfolio strategy.

    Parameters
    ----------
    mu : (d,) array — asset drifts
    sigma : (d,) array — asset volatilities
    r : float — risk-free rate
    gamma : float — CRRA parameter (< 1, ≠ 0)
    """

    def __init__(
        self,
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        r: float,
        gamma: float,
    ):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.r = r
        self.gamma = gamma
        self.n_assets = len(self.mu)

    def optimal_weights(self) -> NDArray[np.float64]:
        """Compute the Merton optimal portfolio weights.

        Returns
        -------
        pi_star : (d,) optimal fraction in each risky asset
        """
        excess = self.mu - self.r
        vol_sq = self.sigma ** 2
        return excess / ((1 - self.gamma) * vol_sq)

    def value_function(
        self,
        W: NDArray[np.float64] | float,
        t: float,
        T: float,
    ) -> NDArray[np.float64] | float:
        """Evaluate the closed-form value function.

        V(W, t) = (W^γ / γ) · exp(A · (T - t))

        Parameters
        ----------
        W : wealth (scalar or array)
        t : current time
        T : terminal time

        Returns
        -------
        V : value function
        """
        gamma = self.gamma
        pi = self.optimal_weights()

        # Compute A coefficient
        excess = self.mu - self.r
        port_excess = np.dot(pi, excess)
        port_vol_sq = np.sum((pi * self.sigma) ** 2)

        A = gamma * (self.r + port_excess - 0.5 * (1 - gamma) * port_vol_sq)

        return (np.power(W, gamma) / gamma) * np.exp(A * (T - t))

    def solve(self, W0: float, T: float) -> MertonSolution:
        """Compute the full Merton solution.

        Parameters
        ----------
        W0 : initial wealth
        T : time horizon

        Returns
        -------
        MertonSolution
        """
        pi_star = self.optimal_weights()
        gamma = self.gamma

        excess = self.mu - self.r
        port_excess = np.dot(pi_star, excess)
        port_vol_sq = np.sum((pi_star * self.sigma) ** 2)

        A = gamma * (self.r + port_excess - 0.5 * (1 - gamma) * port_vol_sq)

        V0 = (W0 ** gamma / gamma) * np.exp(A * T)
        # Certainty equivalent: W_CE such that U(W_CE) = V0
        # W_CE^γ / γ = V0 → W_CE = (γ V0)^(1/γ)
        CE = (gamma * V0) ** (1 / gamma)

        return MertonSolution(
            pi_star=pi_star,
            expected_utility=V0,
            certainty_equivalent=CE,
            value_function_coef=A,
        )

    def simulate_wealth(
        self,
        W0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Simulate wealth paths under the Merton strategy (constant π*).

        Returns
        -------
        wealth : (n_paths, n_steps + 1) array
        """
        dt = T / n_steps
        pi = self.optimal_weights()

        wealth = np.empty((n_paths, n_steps + 1))
        wealth[:, 0] = W0

        for t in range(n_steps):
            Z = rng.standard_normal((n_paths, self.n_assets))
            # Portfolio log-return
            port_drift = self.r + np.dot(pi, self.mu - self.r) - 0.5 * np.sum((pi * self.sigma) ** 2)
            port_vol = np.sqrt(np.sum((pi * self.sigma) ** 2))
            log_ret = port_drift * dt + port_vol * np.sqrt(dt) * Z[:, 0]
            wealth[:, t + 1] = wealth[:, t] * np.exp(log_ret)

        return wealth
