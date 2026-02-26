"""
Multi-asset geometric Brownian motion with regime-dependent drift and volatility.

Vectorised Euler–Maruyama discretisation.  No Python-level loops over paths.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MultiAssetGBM:
    """Correlated GBM dynamics with regime-switching parameters.

    Under regime k the i-th asset follows:
        dS_i / S_i = mu_k[i] dt + sigma_k[i] dW_i

    where the W_i are correlated via the given correlation matrix.

    Parameters
    ----------
    mu : (K, d) array
        Drift vectors per regime.
    sigma : (K, d) array
        Volatility vectors per regime.
    correlation : (d, d) array
        Correlation matrix of the Brownian motions.
    risk_free_rate : float
        Continuously compounded risk-free rate.
    """

    def __init__(
        self,
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        correlation: NDArray[np.float64],
        risk_free_rate: float = 0.03,
    ):
        self.mu = np.asarray(mu, dtype=np.float64)  # (K, d)
        self.sigma = np.asarray(sigma, dtype=np.float64)  # (K, d)
        self.corr = np.asarray(correlation, dtype=np.float64)
        self.r = risk_free_rate
        self.n_regimes, self.n_assets = self.mu.shape

    # ------------------------------------------------------------------
    # Euler–Maruyama step  (vectorised over paths)
    # ------------------------------------------------------------------
    def step(
        self,
        S: NDArray[np.float64],
        regimes: NDArray[np.int64],
        dW: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Advance prices by one Euler–Maruyama step.

        Parameters
        ----------
        S : (n_paths, d) array
            Current asset prices.
        regimes : (n_paths,) int array
            Current regime for each path.
        dW : (n_paths, d) array
            Correlated Brownian increments for this step.
        dt : float
            Time increment.

        Returns
        -------
        S_next : (n_paths, d) array
            Updated asset prices (floored at 1e-12 to avoid negativity).
        """
        mu_k = self.mu[regimes]  # (n_paths, d)
        sigma_k = self.sigma[regimes]  # (n_paths, d)

        drift = mu_k * dt
        diffusion = sigma_k * dW
        # Milstein correction is zero for diagonal volatility
        S_next = S * np.exp((mu_k - 0.5 * sigma_k**2) * dt + diffusion)
        return np.maximum(S_next, 1e-12)

    # ------------------------------------------------------------------
    # Full path simulation
    # ------------------------------------------------------------------
    def simulate_paths(
        self,
        S0: NDArray[np.float64],
        regimes: NDArray[np.int64],
        dW: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Simulate full price paths given regime and noise arrays.

        Parameters
        ----------
        S0 : (d,) array
            Initial prices.
        regimes : (n_paths, n_steps + 1) int array
            Regime at each time point.
        dW : (n_paths, n_steps, d) array
            Pre-generated correlated Brownian increments.
        dt : float
            Time step.

        Returns
        -------
        S : (n_paths, n_steps + 1, d) array
            Simulated price paths.
        """
        n_paths, n_steps, d = dW.shape
        S = np.empty((n_paths, n_steps + 1, d), dtype=np.float64)
        S[:, 0, :] = S0[None, :]

        for t in range(n_steps):
            S[:, t + 1, :] = self.step(S[:, t, :], regimes[:, t], dW[:, t, :], dt)
        return S

    # ------------------------------------------------------------------
    # Log-return computation (useful for calibration)
    # ------------------------------------------------------------------
    @staticmethod
    def log_returns(S: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute log-returns from price paths.

        Parameters
        ----------
        S : (..., T+1, d) array

        Returns
        -------
        log_ret : (..., T, d) array
        """
        return np.log(S[..., 1:, :] / S[..., :-1, :])
