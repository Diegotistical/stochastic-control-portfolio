"""
Actor-Critic networks for portfolio control via Deep RL.

Actor: outputs a Dirichlet distribution over the portfolio simplex.
Critic: outputs a scalar value estimate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


class Actor(nn.Module):
    """Policy network mapping observations to a Dirichlet distribution.

    The Dirichlet ensures the output portfolio weights lie on the simplex.

    Parameters
    ----------
    obs_dim : int
        Observation vector dimension.
    action_dim : int
        Number of risky assets (Dirichlet dimension = action_dim + 1
        for risk-free).
    hidden_sizes : list[int]
        Hidden layer sizes.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | None = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 64]

        # Build MLP
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Tanh())
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        # Output: concentration parameters for Dirichlet (must be > 0)
        # action_dim + 1 to include risk-free asset
        self.concentration_head = nn.Linear(in_dim, action_dim + 1)

    def forward(self, obs: torch.Tensor) -> Dirichlet:
        """Forward pass.

        Parameters
        ----------
        obs : (batch, obs_dim) tensor

        Returns
        -------
        dist : Dirichlet distribution over (action_dim + 1) assets
        """
        features = self.backbone(obs)
        # Softplus to ensure concentration > 0, + 1 for numerical stability
        alpha = F.softplus(self.concentration_head(features)) + 1.0
        return Dirichlet(alpha)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log-probability.

        Parameters
        ----------
        obs : (batch, obs_dim)
        deterministic : bool
            If True, return the mode of the Dirichlet.

        Returns
        -------
        action : (batch, action_dim) — risky asset weights only
        log_prob : (batch,) — log probability of the action
        """
        dist = self.forward(obs)

        if deterministic:
            # Mode of Dirichlet: (alpha_i - 1) / (sum(alpha) - K) for alpha_i > 1
            alpha = dist.concentration
            K = alpha.shape[-1]
            mode = (alpha - 1) / (alpha.sum(dim=-1, keepdim=True) - K).clamp(min=1e-6)
            mode = mode.clamp(min=0)
            mode = mode / mode.sum(dim=-1, keepdim=True)
            action_full = mode
        else:
            action_full = dist.rsample()  # (batch, action_dim + 1)

        log_prob = dist.log_prob(action_full)
        # Return only risky asset weights (drop risk-free)
        action = action_full[..., :-1]
        return action, log_prob

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the policy distribution."""
        dist = self.forward(obs)
        return dist.entropy()


class Critic(nn.Module):
    """Value network mapping observations to a scalar value estimate.

    Parameters
    ----------
    obs_dim : int
    hidden_sizes : list[int]
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list[int] | None = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 64]

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Tanh())
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        obs : (batch, obs_dim)

        Returns
        -------
        value : (batch,) scalar value estimates
        """
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined Actor-Critic model.

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    hidden_sizes : list[int]
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.actor = Actor(obs_dim, action_dim, hidden_sizes)
        self.critic = Critic(obs_dim, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> tuple[Dirichlet, torch.Tensor]:
        """Forward pass returning policy distribution and value."""
        dist = self.actor(obs)
        value = self.critic(obs)
        return dist, value
