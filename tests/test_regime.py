"""Unit tests for HMM regime switching."""

import numpy as np
import pytest

from src.model.regime import HiddenMarkovRegime


class TestHiddenMarkovRegime:
    """Test regime model correctness."""

    @pytest.fixture
    def hmm(self):
        Q = np.array([[-0.5, 0.5], [1.0, -1.0]])
        return HiddenMarkovRegime(Q)

    def test_stationary_distribution(self, hmm):
        """Stationary distribution matches eigenvalue computation.

        For Q = [[-0.5, 0.5], [1.0, -1.0]]:
        π Q = 0, π₁ + π₂ = 1  →  π = [2/3, 1/3]
        """
        pi = hmm.stationary_distribution()
        expected = np.array([2 / 3, 1 / 3])
        np.testing.assert_allclose(pi, expected, atol=1e-10)

    def test_transition_matrix_rows_sum_to_one(self, hmm):
        """P(dt) rows must sum to 1."""
        P = hmm.transition_matrix(dt=0.01)
        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_transition_matrix_nonnegative(self, hmm):
        """P(dt) entries must be non-negative."""
        P = hmm.transition_matrix(dt=0.01)
        assert np.all(P >= 0)

    def test_simulation_shape(self, hmm):
        """Simulated regime array has correct shape."""
        rng = np.random.default_rng(42)
        regimes = hmm.simulate(n_paths=100, n_steps=50, dt=0.01, rng=rng)
        assert regimes.shape == (100, 51)

    def test_simulation_values_valid(self, hmm):
        """Regime values must be 0 or 1 for 2-regime model."""
        rng = np.random.default_rng(42)
        regimes = hmm.simulate(n_paths=1000, n_steps=100, dt=0.01, rng=rng)
        assert np.all((regimes == 0) | (regimes == 1))

    def test_empirical_stationary(self, hmm):
        """Long simulation should produce empirical frequencies near stationary dist."""
        rng = np.random.default_rng(42)
        regimes = hmm.simulate(n_paths=1, n_steps=100_000, dt=0.01, rng=rng)
        freq = np.bincount(regimes[0], minlength=2) / regimes.shape[1]
        expected = hmm.stationary_distribution()
        np.testing.assert_allclose(freq, expected, atol=0.02)
