"""Unit tests for benchmarks and RL environment."""

import numpy as np
import pytest

from src.benchmarks.merton import MertonStrategy
from src.benchmarks.mean_variance import MeanVarianceStrategy
from src.benchmarks.risk_parity import RiskParityStrategy


class TestMertonStrategy:
    def test_analytical_weights(self):
        """π* = (μ−r) / ((1−γ)σ²) for diagonal case."""
        mu = np.array([0.08, 0.12])
        sigma = np.array([0.15, 0.20])
        r = 0.03
        gamma = -2.0

        merton = MertonStrategy(mu, sigma, r, gamma)
        pi = merton.optimal_weights()

        expected = (mu - r) / ((1 - gamma) * sigma**2)
        np.testing.assert_allclose(pi, expected)

    def test_certainty_equivalent_positive(self):
        mu = np.array([0.08])
        sigma = np.array([0.20])
        sol = MertonStrategy(mu, sigma, 0.03, -2.0).solve(W0=1.0, T=1.0)
        assert sol.certainty_equivalent > 0


class TestMeanVariance:
    def test_weights_sum_leq_one(self):
        mu = np.array([0.08, 0.12])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        mvs = MeanVarianceStrategy(mu, cov, r=0.03)
        w = mvs.optimal_weights(allow_short=False)
        assert w.sum() <= 1.0 + 1e-6
        assert np.all(w >= -1e-6)


class TestRiskParity:
    def test_equal_risk_contribution(self):
        """Risk contributions should be approximately equal."""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        rp = RiskParityStrategy(cov)
        w = rp.optimal_weights()

        assert abs(w.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {w.sum()}"

        rc = rp.risk_contributions(w)
        # Each asset should contribute ~0.5 of total risk
        np.testing.assert_allclose(rc, 0.5, atol=0.05)

    def test_weights_positive(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        rp = RiskParityStrategy(cov)
        w = rp.optimal_weights()
        assert np.all(w > 0)
