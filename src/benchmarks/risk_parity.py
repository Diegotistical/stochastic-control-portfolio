"""
Risk-Parity portfolio strategy.

Equal risk contribution: each asset contributes equally to total
portfolio risk. Solved via iterative optimisation.

    σ_i · π_i · (Σπ)_i / (π^T Σ π) = 1/d   for all i
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


class RiskParityStrategy:
    """Risk-Parity (Equal Risk Contribution) portfolio.

    Parameters
    ----------
    cov : (d, d) covariance matrix
    """

    def __init__(self, cov: NDArray[np.float64]):
        self.cov = np.asarray(cov, dtype=np.float64)
        self.n_assets = self.cov.shape[0]

    def optimal_weights(self) -> NDArray[np.float64]:
        """Compute risk-parity weights via optimisation.

        Minimises the sum of squared deviations of marginal risk
        contributions from the equal-contribution target.

        Returns
        -------
        pi : (d,) risk-parity weights (sum to 1)
        """
        d = self.n_assets
        target_rc = 1.0 / d

        def objective(w):
            w = np.maximum(w, 1e-10)
            sigma_p = np.sqrt(w @ self.cov @ w)
            mrc = (self.cov @ w) / sigma_p  # marginal risk contribution
            rc = w * mrc / sigma_p  # risk contribution fraction
            return np.sum((rc - target_rc) ** 2)

        # Start from equal weight
        x0 = np.ones(d) / d
        bounds = [(1e-6, 1)] * d
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

        result = minimize(
            objective, x0, bounds=bounds,
            constraints=constraints, method="SLSQP"
        )

        return result.x

    def risk_contributions(
        self, w: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute risk contribution of each asset.

        Parameters
        ----------
        w : (d,) portfolio weights

        Returns
        -------
        rc : (d,) fraction of total risk from each asset
        """
        sigma_p = np.sqrt(w @ self.cov @ w)
        mrc = (self.cov @ w) / sigma_p
        rc = w * mrc / sigma_p
        return rc

    @classmethod
    def from_returns(
        cls,
        returns: NDArray[np.float64],
        annualise: float = 252.0,
    ) -> RiskParityStrategy:
        """Construct from historical return data.

        Parameters
        ----------
        returns : (T, d) array
        annualise : periods per year

        Returns
        -------
        RiskParityStrategy
        """
        cov = np.cov(returns.T) * annualise
        return cls(cov=cov)
