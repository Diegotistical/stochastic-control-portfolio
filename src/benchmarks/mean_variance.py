"""
Mean-Variance (Markowitz) portfolio strategy.

Static allocation via quadratic programming, plus a rolling-window
dynamic variant for comparison.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


class MeanVarianceStrategy:
    """Markowitz Mean-Variance portfolio optimisation.

    Parameters
    ----------
    mu : (d,) expected returns
    cov : (d, d) covariance matrix
    r : float — risk-free rate
    gamma : float — risk aversion (controls mean-variance trade-off)
    """

    def __init__(
        self,
        mu: NDArray[np.float64],
        cov: NDArray[np.float64],
        r: float = 0.03,
        gamma: float = 2.0,
    ):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.r = r
        self.gamma = gamma  # risk aversion in MV sense (positive)
        self.n_assets = len(self.mu)

    def optimal_weights(self, allow_short: bool = False) -> NDArray[np.float64]:
        """Compute optimal mean-variance weights.

        Maximise: π^T μ_excess - (γ/2) π^T Σ π

        Parameters
        ----------
        allow_short : bool
            If True, allow negative weights (shorting).

        Returns
        -------
        pi : (d,) optimal risky weights
        """
        excess = self.mu - self.r

        if allow_short:
            # Closed-form: π* = (1/γ) Σ^{-1} (μ − r)
            cov_inv = np.linalg.inv(self.cov)
            return (1 / self.gamma) * cov_inv @ excess

        # Constrained: no shorting, sum of weights ≤ 1
        def neg_utility(w):
            return -(w @ excess - 0.5 * self.gamma * w @ self.cov @ w)

        bounds = [(0, 1)] * self.n_assets
        constraints = [{"type": "ineq", "fun": lambda w: 1 - w.sum()}]
        x0 = np.ones(self.n_assets) / (self.n_assets + 1)

        result = minimize(
            neg_utility, x0, bounds=bounds, constraints=constraints, method="SLSQP"
        )
        return result.x

    def efficient_frontier(
        self, n_points: int = 50
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Compute the efficient frontier.

        Returns
        -------
        returns : (n_points,) expected portfolio returns
        vols : (n_points,) portfolio volatilities
        weights : (n_points, d) optimal weights at each point
        """
        excess = self.mu - self.r

        # Range of target returns
        min_ret = self.r
        max_ret = self.mu.max()
        targets = np.linspace(min_ret, max_ret, n_points)

        vols = np.empty(n_points)
        weights = np.empty((n_points, self.n_assets))

        for i, target in enumerate(targets):

            def portfolio_var(w):
                return w @ self.cov @ w

            constraints = [
                {
                    "type": "eq",
                    "fun": lambda w, t=target: w @ self.mu + (1 - w.sum()) * self.r - t,
                },
            ]
            bounds = [(0, 1)] * self.n_assets
            x0 = np.ones(self.n_assets) / (self.n_assets + 1)

            result = minimize(
                portfolio_var,
                x0,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
            )
            weights[i] = result.x
            vols[i] = np.sqrt(result.fun)

        returns = targets
        return returns, vols, weights

    @classmethod
    def from_returns(
        cls,
        returns: NDArray[np.float64],
        r: float = 0.03,
        gamma: float = 2.0,
        annualise: float = 252.0,
    ) -> MeanVarianceStrategy:
        """Construct from historical return data.

        Parameters
        ----------
        returns : (T, d) array of periodic returns
        r : risk-free rate (annualised)
        gamma : risk aversion
        annualise : number of periods per year

        Returns
        -------
        MeanVarianceStrategy
        """
        mu = returns.mean(axis=0) * annualise
        cov = np.cov(returns.T) * annualise
        return cls(mu=mu, cov=cov, r=r, gamma=gamma)
