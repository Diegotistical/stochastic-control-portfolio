"""
Wonham filter for latent regime estimation under partial information.

Implements the continuous-time Wonham filtering SDE discretised via
Euler's method to track posterior regime probabilities given price obs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.exceptions import FilterDegeneracyError


class WonhamFilter:
    """Discrete-time approximation of the Wonham filter.

    The Wonham filter tracks the posterior probability of each regime
    given observed asset returns.  For K regimes:

        dp_i = (Q^T p)_i dt + p_i (h_i - p . h) . dν

    where:
        h_i = Σ_inv @ mu_i   (signal-to-noise ratio per regime)
        dν = Σ^{-1/2} (dX - p . μ dt)   (innovation process)

    We discretise this and project back onto the probability simplex
    at each step to maintain numerical stability.

    Parameters
    ----------
    Q : (K, K) array
        Generator matrix of the Markov chain.
    mu : (K, d) array
        Regime-dependent drift vectors.
    sigma : (K, d) array
        Regime-dependent volatility vectors.
    correlation : (d, d) array
        Correlation matrix of the Brownian motions.
    """

    def __init__(
        self,
        Q: NDArray[np.float64],
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        correlation: NDArray[np.float64],
    ):
        self.Q = np.asarray(Q, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.n_regimes, self.n_assets = self.mu.shape

        # Build covariance and its inverse
        # Σ = diag(σ_avg) @ corr @ diag(σ_avg)  — we use average vol across regimes
        sigma_avg = self.sigma.mean(axis=0)
        self.Sigma = np.diag(sigma_avg) @ correlation @ np.diag(sigma_avg)
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.Sigma_inv_sqrt = np.linalg.cholesky(self.Sigma_inv)

        # Signal-to-noise: h_k = Sigma_inv @ mu_k, shape (K, d)
        self.h = (self.Sigma_inv @ self.mu.T).T  # (K, d)

    def filter(
        self,
        log_returns: NDArray[np.float64],
        dt: float,
        p0: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Apply the Wonham filter to a sequence of log-returns.

        Parameters
        ----------
        log_returns : (n_paths, n_steps, d) or (n_steps, d) array
            Observed log-returns (dX ≈ log(S_{t+1}/S_t)).
        dt : float
            Time step size.
        p0 : (K,) array, optional
            Initial belief state.  Defaults to uniform.

        Returns
        -------
        beliefs : (..., n_steps + 1, K) array
            Posterior regime probabilities at each time step.
        """
        squeeze = False
        if log_returns.ndim == 2:
            log_returns = log_returns[None, :, :]
            squeeze = True

        n_paths, n_steps, d = log_returns.shape
        K = self.n_regimes

        if p0 is None:
            p0 = np.ones(K) / K

        beliefs = np.empty((n_paths, n_steps + 1, K), dtype=np.float64)
        beliefs[:, 0, :] = p0[None, :]

        for t in range(n_steps):
            p = beliefs[:, t, :]  # (n_paths, K)
            dX = log_returns[:, t, :]  # (n_paths, d)

            # Posterior-weighted drift
            mu_bar = p @ self.mu  # (n_paths, d)

            # Innovation: dν = dX - mu_bar * dt
            innovation = dX - mu_bar * dt  # (n_paths, d)

            # Transition term: Q^T @ p^T → (K, n_paths) → transpose
            transition = (self.Q.T @ p.T).T  # (n_paths, K)

            # Signal term per regime: p_i * (h_i - sum_j p_j h_j) . innovation
            h_bar = p @ self.h  # (n_paths, d)
            # h_diff[path, k, :] = h[k, :] - h_bar[path, :]
            h_diff = self.h[None, :, :] - h_bar[:, None, :]  # (n_paths, K, d)
            # signal[path, k] = p[path, k] * h_diff[path, k, :] @ innovation[path, :]
            signal = p[:, :, None] * h_diff  # (n_paths, K, d)
            signal_dot_innov = np.einsum("nkd,nd->nk", signal, innovation)

            # Euler update
            p_new = p + transition * dt + signal_dot_innov

            # Project onto simplex (clip + normalise)
            p_new = self._project_simplex(p_new)
            beliefs[:, t + 1, :] = p_new

        if squeeze:
            beliefs = beliefs[0]
        return beliefs

    def posterior_params(
        self,
        beliefs: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute posterior-averaged drift and volatility given beliefs.

        Parameters
        ----------
        beliefs : (..., K) array
            Current posterior regime probabilities.

        Returns
        -------
        mu_post : (..., d) array
            Posterior-weighted drift.
        sigma_post : (..., d) array
            Posterior-weighted volatility.
        """
        mu_post = beliefs @ self.mu
        sigma_post = beliefs @ self.sigma
        return mu_post, sigma_post

    @staticmethod
    def _project_simplex(p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project onto the probability simplex via clipping and renormalisation.

        Parameters
        ----------
        p : (..., K) array

        Returns
        -------
        p_proj : (..., K) array on the simplex.
        """
        p = np.clip(p, 1e-15, None)
        return p / p.sum(axis=-1, keepdims=True)
