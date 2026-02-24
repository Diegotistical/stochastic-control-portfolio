"""
PPO training loop for the portfolio environment.

Implements Proximal Policy Optimization with:
  - GAE (Generalized Advantage Estimation)
  - Entropy regularisation
  - Gradient clipping
  - Checkpoint saving
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from src.solver.actor_critic import ActorCritic
from src.solver.rl_env import VectorizedPortfolioEnv

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training statistics."""
    episode_rewards: list[float] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    mean_wealth: list[float] = field(default_factory=list)


class PPOTrainer:
    """Proximal Policy Optimization trainer.

    Parameters
    ----------
    env : VectorizedPortfolioEnv
    model : ActorCritic
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    clip_epsilon : float
        PPO clipping parameter.
    entropy_coef : float
        Entropy bonus coefficient.
    value_coef : float
        Value loss coefficient.
    n_epochs : int
        PPO update epochs per rollout.
    max_grad_norm : float
        Gradient clipping norm.
    gae_lambda : float
        GAE lambda parameter.
    checkpoint_dir : str
        Directory for saving checkpoints.
    """

    def __init__(
        self,
        env: VectorizedPortfolioEnv,
        model: ActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_epochs: int = 4,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        checkpoint_dir: str = "checkpoints/",
    ):
        self.env = env
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.metrics = TrainingMetrics()

    def train(self, n_episodes: int, rollout_length: int | None = None) -> TrainingMetrics:
        """Run PPO training.

        Parameters
        ----------
        n_episodes : int
            Total number of training episodes.
        rollout_length : int, optional
            Steps per rollout. Defaults to env's n_steps.

        Returns
        -------
        TrainingMetrics
        """
        if rollout_length is None:
            rollout_length = self.env.envs[0].n_steps

        n_envs = self.env.n_envs
        total_updates = n_episodes // n_envs

        for update in range(total_updates):
            # Collect rollout
            rollout = self._collect_rollout(rollout_length)

            # Compute advantages via GAE
            advantages, returns = self._compute_gae(rollout)

            # PPO update
            policy_loss, value_loss, entropy = self._ppo_update(
                rollout, advantages, returns
            )

            # Track metrics
            mean_reward = rollout["rewards"].sum(axis=1).mean()
            mean_wealth = np.mean([info.get("wealth", 1.0) for info in rollout["infos"][-1]])

            self.metrics.episode_rewards.append(float(mean_reward))
            self.metrics.policy_losses.append(float(policy_loss))
            self.metrics.value_losses.append(float(value_loss))
            self.metrics.entropies.append(float(entropy))
            self.metrics.mean_wealth.append(float(mean_wealth))

            if update % 100 == 0:
                logger.info(
                    f"Update {update}/{total_updates}: "
                    f"reward={mean_reward:.4f}, wealth={mean_wealth:.4f}, "
                    f"π_loss={policy_loss:.4f}, v_loss={value_loss:.4f}"
                )

            if update % 500 == 0 and update > 0:
                self._save_checkpoint(update)

        self._save_checkpoint(total_updates)
        return self.metrics

    def _collect_rollout(self, length: int) -> dict:
        """Collect experience from the vectorised environment."""
        n_envs = self.env.n_envs
        obs_list, action_list, reward_list = [], [], []
        log_prob_list, value_list, done_list, info_list = [], [], [], []

        obs = self.env.reset()

        for t in range(length):
            obs_t = torch.FloatTensor(obs).to(self.device)

            with torch.no_grad():
                dist, value = self.model(obs_t)
                action_full = dist.sample()
                log_prob = dist.log_prob(action_full)
                action = action_full[..., :-1].cpu().numpy()  # risky weights only
                value = value.cpu().numpy()
                log_prob = log_prob.cpu().numpy()

            next_obs, rewards, dones, infos = self.env.step(action)

            obs_list.append(obs)
            action_list.append(action_full.cpu().numpy())
            reward_list.append(rewards)
            log_prob_list.append(log_prob)
            value_list.append(value)
            done_list.append(dones)
            info_list.append(infos)

            obs = next_obs

        # Final value for bootstrapping
        with torch.no_grad():
            final_value = self.model.critic(
                torch.FloatTensor(obs).to(self.device)
            ).cpu().numpy()

        return {
            "obs": np.array(obs_list),           # (T, n_envs, obs_dim)
            "actions": np.array(action_list),     # (T, n_envs, action_dim+1)
            "rewards": np.array(reward_list),     # (T, n_envs)
            "log_probs": np.array(log_prob_list), # (T, n_envs)
            "values": np.array(value_list),       # (T, n_envs)
            "dones": np.array(done_list),         # (T, n_envs)
            "final_value": final_value,           # (n_envs,)
            "infos": info_list,
        }

    def _compute_gae(
        self, rollout: dict
    ) -> tuple[NDArray, NDArray]:
        """Compute Generalized Advantage Estimation."""
        T = rollout["rewards"].shape[0]
        n_envs = rollout["rewards"].shape[1]

        advantages = np.zeros((T, n_envs))
        returns = np.zeros((T, n_envs))

        last_gae = 0
        last_value = rollout["final_value"]

        for t in reversed(range(T)):
            mask = 1 - rollout["dones"][t]
            delta = (
                rollout["rewards"][t]
                + self.gamma * last_value * mask
                - rollout["values"][t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + rollout["values"][t]
            last_value = rollout["values"][t]

        return advantages, returns

    def _ppo_update(
        self,
        rollout: dict,
        advantages: NDArray,
        returns: NDArray,
    ) -> tuple[float, float, float]:
        """Perform PPO clipped objective update."""
        T, n_envs = advantages.shape
        batch_size = T * n_envs

        # Flatten
        obs_flat = torch.FloatTensor(
            rollout["obs"].reshape(batch_size, -1)
        ).to(self.device)
        actions_flat = torch.FloatTensor(
            rollout["actions"].reshape(batch_size, -1)
        ).to(self.device)
        old_log_probs = torch.FloatTensor(
            rollout["log_probs"].reshape(batch_size)
        ).to(self.device)
        adv_flat = torch.FloatTensor(
            advantages.reshape(batch_size)
        ).to(self.device)
        ret_flat = torch.FloatTensor(
            returns.reshape(batch_size)
        ).to(self.device)

        # Normalise advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.n_epochs):
            # Recompute log probs and values
            dist, values = self.model(obs_flat)
            new_log_probs = dist.log_prob(actions_flat)
            entropy = dist.entropy().mean()

            # Policy loss (clipped)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values, ret_flat)

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_pi_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            total_entropy += entropy.item()

        n = self.n_epochs
        return total_pi_loss / n, total_v_loss / n, total_entropy / n

    def _save_checkpoint(self, update: int) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"checkpoint_{update}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update": update,
            "metrics": self.metrics,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint: {path}")
