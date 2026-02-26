"""
HMM calibration from historical return data using Expectation-Maximisation.

Wraps hmmlearn's GaussianHMM for MLE of regime-dependent μ, σ, and
the generator matrix Q.  Provides train/test split helpers and
regime probability comparison with macroeconomic dates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.exceptions import CalibrationError


@dataclass
class CalibrationResult:
    """Container for HMM calibration output."""

    mu: NDArray[np.float64]  # (K, d) regime-dependent means
    sigma: NDArray[np.float64]  # (K, d) regime-dependent vols
    covariances: NDArray[np.float64]  # (K, d, d) full covariance matrices
    generator: NDArray[np.float64]  # (K, K) estimated generator matrix
    transition_matrix: NDArray[np.float64]  # (K, K) one-step transition
    stationary_distribution: NDArray[np.float64]  # (K,)
    log_likelihood: float
    regime_probs: NDArray[np.float64]  # (T, K) smoothed regime probabilities
    n_regimes: int


class HMMCalibrator:
    """Calibrate a Gaussian HMM from historical return data.

    Parameters
    ----------
    n_regimes : int
        Number of hidden regimes.
    n_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance for EM.
    """

    def __init__(self, n_regimes: int = 2, n_iter: int = 200, tol: float = 1e-4):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol

    def fit(
        self,
        returns: NDArray[np.float64] | pd.DataFrame,
        dt: float = 1 / 252,
    ) -> CalibrationResult:
        """Fit HMM to observed return data via EM (Baum–Welch).

        Parameters
        ----------
        returns : (T, d) array or DataFrame
            Historical log-returns.
        dt : float
            Time step (for annualisation and generator estimation).

        Returns
        -------
        CalibrationResult
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise CalibrationError(
                "hmmlearn is required for HMM calibration. "
                "Install via: pip install hmmlearn"
            ) from e

        if isinstance(returns, pd.DataFrame):
            returns = returns.values

        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim == 1:
            returns = returns[:, None]

        T_obs, d = returns.shape

        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=42,
        )

        try:
            model.fit(returns)
        except Exception as e:
            raise CalibrationError(f"EM fitting failed: {e}") from e

        if not model.monitor_.converged:
            raise CalibrationError(
                f"EM did not converge after {self.n_iter} iterations. "
                f"Final log-likelihood change: {model.monitor_.history[-1]:.2e}"
            )

        # Extract parameters
        mu = model.means_  # (K, d)
        covariances = model.covars_  # (K, d, d)
        # Extract diagonal volatilities
        sigma = np.sqrt(
            np.array([np.diag(covariances[k]) for k in range(self.n_regimes)])
        )

        # Transition matrix (one-step, dt-frequency)
        P = model.transmat_  # (K, K)

        # Estimate generator: Q = log(P) / dt  (matrix logarithm)
        Q = self._estimate_generator(P, dt)

        # Stationary distribution
        stat_dist = self._stationary_from_generator(Q)

        # Smoothed regime probabilities
        regime_probs = model.predict_proba(returns)

        return CalibrationResult(
            mu=mu / dt,  # annualise
            sigma=sigma / np.sqrt(dt),  # annualise
            covariances=covariances / dt,
            generator=Q,
            transition_matrix=P,
            stationary_distribution=stat_dist,
            log_likelihood=model.score(returns),
            regime_probs=regime_probs,
            n_regimes=self.n_regimes,
        )

    @staticmethod
    def _estimate_generator(P: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """Estimate CTMC generator from transition matrix via matrix logarithm.

        Q = logm(P) / dt

        Falls back to a simple approximation if logm yields non-real entries.
        """
        from scipy.linalg import logm

        logP = logm(P)
        Q = np.real(logP) / dt

        # Enforce generator structure: off-diag ≥ 0, rows sum to 0
        K = Q.shape[0]
        for i in range(K):
            for j in range(K):
                if i != j:
                    Q[i, j] = max(Q[i, j], 1e-10)
            Q[i, i] = -np.sum(Q[i, [j for j in range(K) if j != i]])

        return Q

    @staticmethod
    def _stationary_from_generator(Q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute stationary distribution from generator matrix."""
        K = Q.shape[0]
        A = np.vstack([Q.T, np.ones(K)])
        b = np.zeros(K + 1)
        b[-1] = 1.0
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.clip(pi, 0, None) / np.clip(pi, 0, None).sum()


def download_data(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
    start, end : str
        Date strings (YYYY-MM-DD).

    Returns
    -------
    log_returns : DataFrame with columns = tickers
    """
    import yfinance as yf

    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices = data["Close"] if len(tickers) > 1 else data["Close"].to_frame()
    prices = prices.dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret
