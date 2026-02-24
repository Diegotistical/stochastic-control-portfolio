"""Unit tests for market dynamics (GBM)."""

import numpy as np
import pytest

from src.model.dynamics import MultiAssetGBM
from src.model.noise import cholesky_factor, generate_increments


class TestMultiAssetGBM:
    """Test GBM log-return moments match analytical values."""

    @pytest.fixture
    def gbm(self):
        mu = np.array([[0.08, 0.12]])   # single regime
        sigma = np.array([[0.15, 0.20]])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        return MultiAssetGBM(mu, sigma, corr, risk_free_rate=0.03)

    def test_log_return_mean(self, gbm):
        """E[log(S_{t+1}/S_t)] = (μ - σ²/2) dt."""
        rng = np.random.default_rng(42)
        n_paths, n_steps = 50_000, 1
        dt = 1 / 252
        S0 = np.array([100.0, 100.0])

        chol = cholesky_factor(gbm.corr)
        dW = generate_increments(n_paths, n_steps, 2, dt, chol, rng)
        regimes = np.zeros((n_paths, n_steps + 1), dtype=np.int64)

        S = gbm.simulate_paths(S0, regimes, dW, dt)
        log_ret = MultiAssetGBM.log_returns(S)[:, 0, :]  # (n_paths, d)

        expected_mean = (gbm.mu[0] - 0.5 * gbm.sigma[0] ** 2) * dt
        empirical_mean = log_ret.mean(axis=0)

        np.testing.assert_allclose(empirical_mean, expected_mean, atol=0.002)

    def test_log_return_variance(self, gbm):
        """Var[log(S_{t+1}/S_t)] = σ² dt."""
        rng = np.random.default_rng(123)
        n_paths = 100_000
        dt = 1 / 252
        S0 = np.array([100.0, 100.0])

        chol = cholesky_factor(gbm.corr)
        dW = generate_increments(n_paths, 1, 2, dt, chol, rng)
        regimes = np.zeros((n_paths, 2), dtype=np.int64)

        S = gbm.simulate_paths(S0, regimes, dW, dt)
        log_ret = MultiAssetGBM.log_returns(S)[:, 0, :]

        expected_var = gbm.sigma[0] ** 2 * dt
        empirical_var = log_ret.var(axis=0)

        np.testing.assert_allclose(empirical_var, expected_var, rtol=0.05)

    def test_positive_prices(self, gbm):
        """Simulated prices must remain positive."""
        rng = np.random.default_rng(42)
        chol = cholesky_factor(gbm.corr)
        dW = generate_increments(1000, 252, 2, 1 / 252, chol, rng)
        regimes = np.zeros((1000, 253), dtype=np.int64)
        S0 = np.array([100.0, 100.0])

        S = gbm.simulate_paths(S0, regimes, dW, 1 / 252)
        assert np.all(S > 0)
