"""
Robustness tests under model misspecification.

Train under one parameter set, test under perturbed parameters,
and measure performance degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """Container for robustness test output."""

    perturbation_type: str
    perturbation_levels: NDArray[np.float64]
    baseline_metric: float
    perturbed_metrics: NDArray[np.float64]
    degradation: NDArray[np.float64]  # percentage drop from baseline


def test_drift_misspecification(
    solve_fn,
    evaluate_fn,
    mu_true: NDArray[np.float64],
    perturbation_pcts: NDArray[np.float64] | None = None,
) -> RobustnessResult:
    """Test robustness to drift estimation errors.

    Parameters
    ----------
    solve_fn : callable
        (mu) → policy  — solves the control problem with given drift
    evaluate_fn : callable
        (policy, mu_true) → metric — evaluates policy under true drift
    mu_true : (d,) true drift
    perturbation_pcts : array of perturbation percentages

    Returns
    -------
    RobustnessResult
    """
    if perturbation_pcts is None:
        perturbation_pcts = np.array([-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5])

    baseline_policy = solve_fn(mu_true)
    baseline_metric = evaluate_fn(baseline_policy, mu_true)

    perturbed_metrics = np.empty(len(perturbation_pcts))

    for i, pct in enumerate(perturbation_pcts):
        mu_assumed = mu_true * (1 + pct)
        policy = solve_fn(mu_assumed)
        perturbed_metrics[i] = evaluate_fn(policy, mu_true)
        logger.info(f"Drift perturbation {pct:+.0%}: metric={perturbed_metrics[i]:.6f}")

    degradation = (baseline_metric - perturbed_metrics) / abs(baseline_metric) * 100

    return RobustnessResult(
        perturbation_type="drift",
        perturbation_levels=perturbation_pcts,
        baseline_metric=baseline_metric,
        perturbed_metrics=perturbed_metrics,
        degradation=degradation,
    )


def test_volatility_misspecification(
    solve_fn,
    evaluate_fn,
    sigma_true: NDArray[np.float64],
    perturbation_pcts: NDArray[np.float64] | None = None,
) -> RobustnessResult:
    """Test robustness to volatility estimation errors."""
    if perturbation_pcts is None:
        perturbation_pcts = np.array([-0.3, -0.15, 0.0, 0.15, 0.3, 0.5])

    baseline_policy = solve_fn(sigma_true)
    baseline_metric = evaluate_fn(baseline_policy, sigma_true)

    perturbed_metrics = np.empty(len(perturbation_pcts))

    for i, pct in enumerate(perturbation_pcts):
        sigma_assumed = sigma_true * (1 + pct)
        policy = solve_fn(sigma_assumed)
        perturbed_metrics[i] = evaluate_fn(policy, sigma_true)
        logger.info(f"Vol perturbation {pct:+.0%}: metric={perturbed_metrics[i]:.6f}")

    degradation = (baseline_metric - perturbed_metrics) / abs(baseline_metric) * 100

    return RobustnessResult(
        perturbation_type="volatility",
        perturbation_levels=perturbation_pcts,
        baseline_metric=baseline_metric,
        perturbed_metrics=perturbed_metrics,
        degradation=degradation,
    )


def test_regime_misspecification(
    solve_fn,
    evaluate_fn,
    Q_true: NDArray[np.float64],
    scaling_factors: NDArray[np.float64] | None = None,
) -> RobustnessResult:
    """Test robustness to regime persistence misspecification.

    Scales the off-diagonal elements of Q (faster/slower switching).
    """
    if scaling_factors is None:
        scaling_factors = np.array([0.25, 0.5, 1.0, 2.0, 4.0])

    baseline_policy = solve_fn(Q_true)
    baseline_metric = evaluate_fn(baseline_policy, Q_true)

    perturbed_metrics = np.empty(len(scaling_factors))

    for i, s in enumerate(scaling_factors):
        K = Q_true.shape[0]
        Q_perturbed = Q_true.copy()
        for row in range(K):
            for col in range(K):
                if row != col:
                    Q_perturbed[row, col] = Q_true[row, col] * s
            Q_perturbed[row, row] = -sum(
                Q_perturbed[row, j] for j in range(K) if j != row
            )

        policy = solve_fn(Q_perturbed)
        perturbed_metrics[i] = evaluate_fn(policy, Q_true)
        logger.info(f"Regime scaling {s:.2f}x: metric={perturbed_metrics[i]:.6f}")

    degradation = (baseline_metric - perturbed_metrics) / abs(baseline_metric) * 100

    return RobustnessResult(
        perturbation_type="regime_persistence",
        perturbation_levels=scaling_factors,
        baseline_metric=baseline_metric,
        perturbed_metrics=perturbed_metrics,
        degradation=degradation,
    )
