"""Unit tests for RL environment."""

import numpy as np
import pytest

from src.solver.rl_env import PortfolioEnv


class TestPortfolioEnv:
    @pytest.fixture
    def env(self):
        return PortfolioEnv(
            mu=np.array([[0.08, 0.12], [0.02, 0.04]]),
            sigma=np.array([[0.15, 0.20], [0.25, 0.35]]),
            correlation=np.array([[1.0, 0.5], [0.5, 1.0]]),
            Q=np.array([[-0.5, 0.5], [1.0, -1.0]]),
            r=0.03,
            gamma=-2.0,
            T=1.0,
            n_steps=10,
            transaction_cost=0.001,
            n_assets=2,
            seed=42,
        )

    def test_reset_shape(self, env):
        obs = env.reset()
        assert obs.shape == (env.obs_dim,)

    def test_step_returns_correct_types(self, env):
        env.reset()
        action = np.array([0.3, 0.3])
        obs, reward, done, info = env.step(action)

        assert obs.shape == (env.obs_dim,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_episode_terminates(self, env):
        env.reset()
        for _ in range(20):
            _, _, done, _ = env.step(np.array([0.3, 0.3]))
            if done:
                break
        assert done, "Episode did not terminate after n_steps"

    def test_wealth_stays_positive(self, env):
        env.reset()
        for _ in range(env.n_steps):
            _, _, done, info = env.step(np.array([0.4, 0.4]))
            assert info["wealth"] > 0
            if done:
                break

    def test_beliefs_on_simplex(self, env):
        env.reset()
        for _ in range(env.n_steps):
            _, _, done, info = env.step(np.array([0.3, 0.3]))
            belief = info["belief"]
            assert np.all(belief >= 0)
            np.testing.assert_allclose(belief.sum(), 1.0, atol=1e-8)
            if done:
                break
