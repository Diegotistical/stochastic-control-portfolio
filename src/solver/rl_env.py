"""
Gymnasium-style environment for portfolio allocation.

State:  (wealth, current_weights, belief_state, time_remaining)
Action: target portfolio weights on the simplex
Reward: CRRA utility increment minus transaction costs
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class PortfolioEnv:
    """Reinforcement learning environment for portfolio optimisation.

    Wraps the market model (GBM + regime switching + Wonham filter)
    into a step-based interface compatible with standard RL training loops.

    Parameters
    ----------
    mu : (K, d) array — regime drifts
    sigma : (K, d) array — regime volatilities
    correlation : (d, d) array
    Q : (K, K) array — generator matrix
    r : float — risk-free rate
    gamma : float — CRRA parameter
    T : float — time horizon
    n_steps : int — steps per episode
    transaction_cost : float — proportional
    n_assets : int — number of risky assets
    seed : int — random seed
    """

    def __init__(
        self,
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        correlation: NDArray[np.float64],
        Q: NDArray[np.float64],
        r: float = 0.03,
        gamma: float = -2.0,
        T: float = 1.0,
        n_steps: int = 252,
        transaction_cost: float = 0.001,
        n_assets: int = 2,
        seed: int = 42,
    ):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.corr = np.asarray(correlation, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.r = r
        self.gamma = gamma
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.tc = transaction_cost
        self.n_assets = n_assets
        self.n_regimes = self.mu.shape[0]

        self.chol = np.linalg.cholesky(self.corr)
        from scipy.linalg import expm
        self.P_dt = expm(self.Q * self.dt)

        self.rng = np.random.default_rng(seed)

        # State dimensions
        # wealth (1) + current_weights (d) + belief (K) + time (1)
        self.obs_dim = 1 + n_assets + self.n_regimes + 1
        self.action_dim = n_assets  # target weights (must sum ≤ 1)

        self.reset()

    def reset(self, seed: int | None = None) -> NDArray[np.float64]:
        """Reset the environment to initial state.

        Returns
        -------
        obs : (obs_dim,) initial observation
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.wealth = 1.0
        self.weights = np.zeros(self.n_assets)  # start fully in risk-free
        self.belief = np.ones(self.n_regimes) / self.n_regimes
        self.regime = self.rng.choice(self.n_regimes, p=self.belief)
        self.step_count = 0
        self.done = False

        return self._get_obs()

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, bool, dict[str, Any]]:
        """Take one time step.

        Parameters
        ----------
        action : (d,) target portfolio weights for risky assets.
                 Risk-free weight = 1 - sum(action).

        Returns
        -------
        obs : (obs_dim,) next observation
        reward : float
        done : bool
        info : dict
        """
        if self.done:
            return self._get_obs(), 0.0, True, {}

        # Clip action to valid range
        action = np.clip(action, 0, 1)
        if action.sum() > 1:
            action = action / action.sum()

        # Transaction cost
        tc_paid = self.tc * np.sum(np.abs(action - self.weights)) * self.wealth
        self.wealth -= tc_paid

        # Update weights to target
        old_weights = self.weights.copy()
        self.weights = action.copy()

        # Regime transition
        cum_P = np.cumsum(self.P_dt[self.regime])
        u = self.rng.random()
        self.regime = int(np.searchsorted(cum_P, u))
        self.regime = min(self.regime, self.n_regimes - 1)

        # Asset returns
        mu_k = self.mu[self.regime]
        sigma_k = self.sigma[self.regime]
        Z = self.rng.standard_normal(self.n_assets)
        dW = np.sqrt(self.dt) * (self.chol @ Z)
        log_ret = (mu_k - 0.5 * sigma_k ** 2) * self.dt + sigma_k * dW

        # Portfolio return
        risky_ret = np.exp(log_ret) - 1
        rf_weight = 1 - self.weights.sum()
        portfolio_ret = rf_weight * self.r * self.dt + np.dot(self.weights, risky_ret)
        self.wealth *= (1 + portfolio_ret)
        self.wealth = max(self.wealth, 1e-10)

        # Update belief (simplified Wonham step)
        self._update_belief(log_ret)

        self.step_count += 1
        self.done = self.step_count >= self.n_steps

        # Reward: CRRA utility increment
        if self.done:
            reward = self._utility(self.wealth)
        else:
            reward = 0.0  # sparse reward at terminal time

        info = {
            "wealth": self.wealth,
            "weights": self.weights.copy(),
            "regime": self.regime,
            "belief": self.belief.copy(),
            "tc_paid": tc_paid,
        }

        return self._get_obs(), reward, self.done, info

    def _get_obs(self) -> NDArray[np.float64]:
        """Construct observation vector."""
        time_remaining = 1.0 - self.step_count / self.n_steps
        return np.concatenate([
            [self.wealth],
            self.weights,
            self.belief,
            [time_remaining],
        ])

    def _utility(self, W: float) -> float:
        """CRRA utility."""
        if self.gamma == 0:
            return np.log(max(W, 1e-10))
        return (max(W, 1e-10) ** self.gamma) / self.gamma

    def _update_belief(self, log_ret: NDArray[np.float64]) -> None:
        """Simplified Wonham filter update for one step."""
        # Likelihood of observed return under each regime
        likelihoods = np.zeros(self.n_regimes)
        for k in range(self.n_regimes):
            mu_k = self.mu[k]
            sigma_k = self.sigma[k]
            # Log-normal density (proportional)
            residual = log_ret - (mu_k - 0.5 * sigma_k ** 2) * self.dt
            likelihoods[k] = np.exp(-0.5 * np.sum(residual ** 2 / (sigma_k ** 2 * self.dt + 1e-30)))

        # Bayes' update
        p_pred = self.P_dt.T @ self.belief  # prediction step
        p_update = likelihoods * p_pred
        total = p_update.sum()
        if total > 1e-30:
            self.belief = p_update / total
        else:
            self.belief = np.ones(self.n_regimes) / self.n_regimes

        # Clip for safety
        self.belief = np.clip(self.belief, 1e-10, None)
        self.belief /= self.belief.sum()


class VectorizedPortfolioEnv:
    """Batch of independent PortfolioEnv instances for efficient training.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments.
    **kwargs
        Arguments passed to each PortfolioEnv.
    """

    def __init__(self, n_envs: int, **kwargs):
        self.n_envs = n_envs
        self.envs = [
            PortfolioEnv(seed=kwargs.pop("seed", 42) + i, **kwargs)
            for i in range(n_envs)
        ]
        self.obs_dim = self.envs[0].obs_dim
        self.action_dim = self.envs[0].action_dim

    def reset(self) -> NDArray[np.float64]:
        """Reset all environments. Returns (n_envs, obs_dim)."""
        return np.array([env.reset() for env in self.envs])

    def step(
        self, actions: NDArray[np.float64]
    ) -> tuple[NDArray, NDArray, NDArray, list[dict]]:
        """Step all environments. actions: (n_envs, action_dim)."""
        results = [env.step(actions[i]) for i, env in enumerate(self.envs)]
        obs = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]

        # Auto-reset done environments
        for i, done in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()

        return obs, rewards, dones, infos
