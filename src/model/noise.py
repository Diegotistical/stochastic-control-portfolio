"""
Correlated Wiener process increment generation.

Provides vectorised generation of correlated Brownian increments
via Cholesky decomposition with optional antithetic variates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cholesky_factor(correlation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the lower-triangular Cholesky factor of a correlation matrix.

    Parameters
    ----------
    correlation : (d, d) array
        Symmetric positive-definite correlation matrix.

    Returns
    -------
    L : (d, d) array
        Lower-triangular Cholesky factor such that L @ L^T = correlation.
    """
    return np.linalg.cholesky(correlation)


def generate_increments(
    n_paths: int,
    n_steps: int,
    n_assets: int,
    dt: float,
    chol: NDArray[np.float64],
    rng: np.random.Generator,
    antithetic: bool = False,
) -> NDArray[np.float64]:
    """Generate correlated Wiener increments dW ~ N(0, dt) with given correlation.

    Parameters
    ----------
    n_paths : int
        Number of Monte-Carlo paths. If *antithetic* is True, half are
        antithetic mirrors, so the effective total is still n_paths
        (n_paths must be even).
    n_steps : int
        Number of discrete time steps.
    n_assets : int
        Number of assets (dimension of Brownian motion).
    dt : float
        Time step size.
    chol : (n_assets, n_assets) array
        Lower-triangular Cholesky factor of the correlation matrix.
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    antithetic : bool, default False
        If True, use antithetic variates for variance reduction.

    Returns
    -------
    dW : (n_paths, n_steps, n_assets) array
        Correlated Brownian increments.
    """
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when using antithetic variates")
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps, n_assets))
        Z = np.concatenate([Z, -Z], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps, n_assets))

    # Correlate: dW = sqrt(dt) * Z @ L^T  (broadcast over paths and steps)
    dW = np.sqrt(dt) * np.einsum("...j,kj->...k", Z, chol)
    return dW
