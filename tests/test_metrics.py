"""Unit tests for performance metrics."""

import numpy as np
import pytest

from src.backtest.metrics import compute_metrics


class TestMetrics:
    def test_sharpe_on_near_constant(self):
        """Near-constant daily return → very high Sharpe."""
        T = 252
        rng = np.random.default_rng(42)
        returns = 0.001 + rng.normal(0, 1e-6, T)  # ~0.1%/day + tiny noise
        wealth = np.cumprod(np.concatenate([[1], 1 + returns]))

        m = compute_metrics(returns, wealth)
        # Ann return ~ 0.001 * 252 = 0.252
        assert abs(m.annualised_return - 0.252) < 0.01
        assert m.sharpe_ratio > 10  # very high for near-deterministic returns

    def test_max_drawdown(self):
        """Known drawdown scenario."""
        wealth = np.array([1.0, 1.2, 1.1, 0.8, 0.9, 1.0])
        returns = np.diff(wealth) / wealth[:-1]

        m = compute_metrics(returns, wealth)
        # Max drawdown: peak=1.2, trough=0.8 → dd = 0.4/1.2 = 33.3%
        assert abs(m.max_drawdown - 1 / 3) < 0.01

    def test_zero_returns(self):
        """All-zero returns should not crash."""
        returns = np.zeros(100)
        wealth = np.ones(101)
        m = compute_metrics(returns, wealth)
        assert m.annualised_return == 0
        assert m.final_wealth == 1.0
