"""
Sensitivity analysis: grid sweep over key parameters.

Sweeps over risk aversion (γ), volatility (σ), regime persistence (Q),
transaction costs (ε), and ambiguity aversion (θ).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product as iterproduct
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Container for sensitivity analysis output."""
    param_names: list[str]
    param_values: dict[str, NDArray[np.float64]]
    metric_name: str
    metric_values: NDArray[np.float64]  # shape depends on # swept params


def run_sensitivity_1d(
    param_name: str,
    param_values: NDArray[np.float64],
    run_fn: Callable[[float], float],
    metric_name: str = "expected_utility",
) -> SensitivityResult:
    """Run 1D sensitivity sweep.

    Parameters
    ----------
    param_name : str
        Name of parameter being swept.
    param_values : (N,) array
        Values to sweep.
    run_fn : callable
        Function mapping param_value → scalar metric.
    metric_name : str

    Returns
    -------
    SensitivityResult
    """
    metrics = np.empty(len(param_values))
    for i, val in enumerate(param_values):
        logger.info(f"Sensitivity {param_name}={val:.4f}")
        metrics[i] = run_fn(val)

    return SensitivityResult(
        param_names=[param_name],
        param_values={param_name: param_values},
        metric_name=metric_name,
        metric_values=metrics,
    )


def run_sensitivity_2d(
    param1_name: str,
    param1_values: NDArray[np.float64],
    param2_name: str,
    param2_values: NDArray[np.float64],
    run_fn: Callable[[float, float], float],
    metric_name: str = "expected_utility",
) -> SensitivityResult:
    """Run 2D sensitivity sweep (heatmap).

    Parameters
    ----------
    run_fn : callable
        (param1_val, param2_val) → scalar metric.

    Returns
    -------
    SensitivityResult with metric_values shape (N1, N2)
    """
    N1 = len(param1_values)
    N2 = len(param2_values)
    metrics = np.empty((N1, N2))

    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            logger.info(f"Sensitivity {param1_name}={v1:.4f}, {param2_name}={v2:.4f}")
            metrics[i, j] = run_fn(v1, v2)

    return SensitivityResult(
        param_names=[param1_name, param2_name],
        param_values={param1_name: param1_values, param2_name: param2_values},
        metric_name=metric_name,
        metric_values=metrics,
    )


# --- Common parameter grids ---

def default_gamma_grid() -> NDArray[np.float64]:
    """Risk aversion parameter sweep."""
    return np.array([-0.5, -1.0, -2.0, -3.0, -5.0, -8.0])


def default_tc_grid() -> NDArray[np.float64]:
    """Transaction cost sweep."""
    return np.array([0.0, 0.0005, 0.001, 0.002, 0.005, 0.01])


def default_theta_grid() -> NDArray[np.float64]:
    """Ambiguity aversion sweep."""
    return np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0])


def default_vol_multiplier_grid() -> NDArray[np.float64]:
    """Volatility scaling factor sweep."""
    return np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
