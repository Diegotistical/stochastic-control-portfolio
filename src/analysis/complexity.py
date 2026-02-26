"""
Computational complexity analysis: PDE vs RL scaling.

Demonstrates the curse of dimensionality for finite-difference HJB
solvers and the polynomial scaling of Deep RL.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ComplexityResult:
    """Container for complexity analysis."""

    dimensions: NDArray[np.int64]
    pde_grid_sizes: NDArray[np.float64]  # total grid points
    pde_memory_mb: NDArray[np.float64]  # estimated memory usage
    pde_flops: NDArray[np.float64]  # estimated FLOPs per step
    rl_params: NDArray[np.float64]  # neural network parameters
    rl_flops_per_step: NDArray[np.float64]  # forward pass FLOPs


def analyze_scaling(
    dimensions: NDArray[np.int64] | None = None,
    n_grid_per_dim: int = 50,
    n_time_steps: int = 200,
    hidden_sizes: list[int] | None = None,
) -> ComplexityResult:
    """Analyze computational scaling of PDE vs RL solvers.

    Parameters
    ----------
    dimensions : array of asset dimensions to analyze
    n_grid_per_dim : grid points per spatial dimension (PDE)
    n_time_steps : time steps
    hidden_sizes : RL network hidden layer sizes

    Returns
    -------
    ComplexityResult
    """
    if dimensions is None:
        dimensions = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20])
    if hidden_sizes is None:
        hidden_sizes = [128, 128, 64]

    n_dims = len(dimensions)
    pde_grid = np.empty(n_dims)
    pde_memory = np.empty(n_dims)
    pde_flops = np.empty(n_dims)
    rl_params_arr = np.empty(n_dims)
    rl_flops_arr = np.empty(n_dims)

    for i, d in enumerate(dimensions):
        # PDE: state space = wealth × d portfolio weights × belief
        # Grid points: N^(d+1) (wealth × d alloc dims, simplified to N^d for allocation)
        # Plus 1 belief dimension
        total_grid = float(n_grid_per_dim) ** (
            d + 1
        )  # wealth + d-1 allocations + belief

        pde_grid[i] = total_grid
        # Memory: 8 bytes per float64 × grid × (V + π + operators)
        pde_memory[i] = total_grid * 8.0 * (1 + d + 3) / 1e6  # MB
        # FLOPs per time step: O(N^d × N^d) for sparse solve, approximately O(N^{d+1})
        pde_flops[i] = total_grid * float(n_grid_per_dim) * n_time_steps

        # RL: observation = (W, π_current, belief, t) → dim = 1 + d + 2 + 1
        obs_dim = d + 4
        # Action = d weights
        action_dim = d

        # Count parameters
        n_params = 0
        in_size = obs_dim
        for h in hidden_sizes:
            n_params += in_size * h + h  # weights + bias
            in_size = h
        n_params += in_size * (action_dim + 1) + (action_dim + 1)  # actor head
        n_params += in_size * 1 + 1  # critic head
        n_params *= 2  # actor + critic separate backbones

        rl_params_arr[i] = n_params
        # FLOPs per forward pass ≈ 2 × params (multiply-add)
        rl_flops_arr[i] = 2 * n_params

        logger.info(
            f"d={d:2d}: PDE grid={total_grid:.2e}, "
            f"PDE memory={pde_memory[i]:.1f}MB, "
            f"RL params={n_params:.0f}"
        )

    return ComplexityResult(
        dimensions=dimensions,
        pde_grid_sizes=pde_grid,
        pde_memory_mb=pde_memory,
        pde_flops=pde_flops,
        rl_params=rl_params_arr,
        rl_flops_per_step=rl_flops_arr,
    )


def format_complexity_table(result: ComplexityResult) -> str:
    """Format complexity analysis as a markdown table.

    Returns
    -------
    table : str — markdown-formatted table
    """
    lines = [
        "| d (assets) | PDE Grid Pts | PDE Memory (MB) | PDE FLOPs/step | RL Params | RL FLOPs/step |",
        "|---|---|---|---|---|---|",
    ]
    for i, d in enumerate(result.dimensions):
        lines.append(
            f"| {d} | {result.pde_grid_sizes[i]:.2e} | "
            f"{result.pde_memory_mb[i]:.1f} | {result.pde_flops[i]:.2e} | "
            f"{result.rl_params[i]:.0f} | {result.rl_flops_per_step[i]:.2e} |"
        )
    return "\n".join(lines)
