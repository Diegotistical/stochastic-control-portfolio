"""Unit tests for the Wonham filter."""

import numpy as np
import pytest

from src.model.wonham_filter import WonhamFilter
from src.model.dynamics import MultiAssetGBM
from src.model.regime import HiddenMarkovRegime
from src.model.noise import cholesky_factor, generate_increments


class TestWonhamFilter:
    """Test filter convergence and simplex preservation."""

    @pytest.fixture
    def setup(self):
        Q = np.array([[-0.5, 0.5], [1.0, -1.0]])
        mu = np.array([[0.10, 0.15], [0.02, 0.03]])
        sigma = np.array([[0.15, 0.20], [0.30, 0.40]])
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        return Q, mu, sigma, corr

    def test_beliefs_on_simplex(self, setup):
        """Beliefs must remain on probability simplex at all times."""
        Q, mu, sigma, corr = setup
        wf = WonhamFilter(Q, mu, sigma, corr)

        rng = np.random.default_rng(42)
        dt = 1 / 252
        n_steps = 500
        log_ret = rng.normal(0, 0.01, (n_steps, 2))

        beliefs = wf.filter(log_ret, dt)

        # Non-negative
        assert np.all(beliefs >= 0), "Beliefs contain negative values"
        # Sum to 1
        np.testing.assert_allclose(beliefs.sum(axis=-1), 1.0, atol=1e-10)

    def test_filter_converges_to_true_regime(self, setup):
        """With strong signal, filter should identify the true regime."""
        Q, mu, sigma, corr = setup

        # Generate data from regime 0 (bull) only
        rng = np.random.default_rng(42)
        dt = 1 / 252
        n_steps = 2000
        # True data from regime 0
        log_ret = (mu[0] - 0.5 * sigma[0] ** 2) * dt + sigma[0] * np.sqrt(dt) * rng.standard_normal((n_steps, 2))

        wf = WonhamFilter(Q, mu, sigma, corr)
        beliefs = wf.filter(log_ret, dt, p0=np.array([0.5, 0.5]))

        # After many observations from regime 0, belief in regime 0 should be high
        final_belief = beliefs[-1, 0]
        assert final_belief > 0.6, f"Filter did not converge: P(regime 0) = {final_belief:.3f}"

    def test_posterior_params_shape(self, setup):
        """Posterior-averaged params should have correct dimensions."""
        Q, mu, sigma, corr = setup
        wf = WonhamFilter(Q, mu, sigma, corr)

        beliefs = np.array([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]])
        mu_post, sigma_post = wf.posterior_params(beliefs)

        assert mu_post.shape == (3, 2)
        assert sigma_post.shape == (3, 2)
