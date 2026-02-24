"""
Hidden Markov regime-switching model.

Continuous-time Markov chain (CTMC) for latent economic regimes.
Provides simulation, stationary distribution, and transition probabilities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


class HiddenMarkovRegime:
    """Continuous-time hidden Markov regime model.

    Parameters
    ----------
    generator : (K, K) array
        Generator (rate) matrix Q.  Rows sum to zero; off-diagonals ≥ 0.
    initial_distribution : (K,) array
        Probability vector for the initial regime.
    """

    def __init__(
        self,
        generator: NDArray[np.float64],
        initial_distribution: NDArray[np.float64] | None = None,
    ):
        self.Q = np.asarray(generator, dtype=np.float64)
        self.n_regimes = self.Q.shape[0]
        if initial_distribution is None:
            self.p0 = self.stationary_distribution()
        else:
            self.p0 = np.asarray(initial_distribution, dtype=np.float64)

    # ------------------------------------------------------------------
    # Stationary distribution
    # ------------------------------------------------------------------
    def stationary_distribution(self) -> NDArray[np.float64]:
        """Compute the stationary distribution π such that π Q = 0, Σπ = 1.

        Solved via the left null-space of Q with the normalisation constraint.
        """
        A = np.vstack([self.Q.T, np.ones(self.n_regimes)])
        b = np.zeros(self.n_regimes + 1)
        b[-1] = 1.0
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        return pi

    # ------------------------------------------------------------------
    # Transition matrix over finite interval
    # ------------------------------------------------------------------
    def transition_matrix(self, dt: float) -> NDArray[np.float64]:
        """Compute P(dt) = exp(Q * dt), the transition probability matrix."""
        return expm(self.Q * dt)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Simulate regime paths via the CTMC.

        Uses the uniformisation / matrix-exponential approach:
        at each step, draw the next regime from the row of P(dt).

        Parameters
        ----------
        n_paths : int
        n_steps : int
        dt : float
        rng : numpy.random.Generator

        Returns
        -------
        regimes : (n_paths, n_steps + 1) int array
            Regime index at each time point (including t=0).
        """
        P = self.transition_matrix(dt)
        cum_P = np.cumsum(P, axis=1)  # (K, K) cumulative transition probs

        regimes = np.empty((n_paths, n_steps + 1), dtype=np.int64)
        # Initial regime
        regimes[:, 0] = rng.choice(self.n_regimes, size=n_paths, p=self.p0)

        for t in range(n_steps):
            u = rng.random(n_paths)
            current = regimes[:, t]
            # Vectorised: for each path, find first column where cumP > u
            # cum_P[current] has shape (n_paths, K)
            regimes[:, t + 1] = (u[:, None] < cum_P[current]).argmax(axis=1)

        return regimes
