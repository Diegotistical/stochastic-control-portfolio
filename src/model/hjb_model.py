"""
Hamilton–Jacobi–Bellman model for CRRA portfolio optimisation.

Encapsulates:
  - Terminal condition (CRRA utility of wealth)
  - Hamiltonian evaluation and optimisation over portfolio weights
  - Transaction cost penalty
  - No-trade region detection
  - Robust (ambiguity-averse) variant via Hansen–Sargent penalty

The HJB equation in log-wealth x = log(W) with belief p:

  V_t + sup_π { A(π, p) V_x + ½ B(π, p) V_xx + C(p) V_p + ½ D(p) V_pp }
      − TC(π, π_prev) = 0

where:
  A(π, p) = r + π·(μ̄(p) − r) − ½ |π·σ̄(p)|²     (drift in log-W)
  B(π, p) = |π·σ̄(p)|²                              (diffusion in log-W)
  C(p)    = Wonham drift of belief
  D(p)    = Wonham diffusion of belief
  TC      = ε |Δπ| (proportional transaction cost)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, minimize

from src.model.grid import ProductGrid


@dataclass
class HJBParams:
    """Parameters for the HJB model."""
    gamma: float                          # CRRA risk aversion
    r: float                              # risk-free rate
    mu: NDArray[np.float64]              # (K, d) regime drifts
    sigma: NDArray[np.float64]           # (K, d) regime vols
    correlation: NDArray[np.float64]     # (d, d)
    Q: NDArray[np.float64]              # (K, K) generator matrix
    transaction_cost: float              # proportional
    # Robust control
    theta: float = 0.0                   # ambiguity parameter (0 = standard)


class HJBModel:
    """HJB model for the stochastic control problem on the product grid.

    Parameters
    ----------
    params : HJBParams
    grid : ProductGrid
    """

    def __init__(self, params: HJBParams, grid: ProductGrid):
        self.params = params
        self.grid = grid
        self.gamma = params.gamma
        self.r = params.r
        self.n_assets = params.mu.shape[1]

        # Precompute Cholesky of correlation
        self.chol = np.linalg.cholesky(params.correlation)

    # ------------------------------------------------------------------
    # Terminal condition
    # ------------------------------------------------------------------
    def terminal_condition(self) -> NDArray[np.float64]:
        """CRRA utility at terminal time: U(W) = W^γ / γ.

        Returns V on the product grid as a flat array.
        """
        W = self.grid.W_mesh  # (Nx, Np)
        gamma = self.gamma
        V = np.power(W, gamma) / gamma
        return self.grid.to_flat(V)

    # ------------------------------------------------------------------
    # Posterior-averaged parameters at each belief grid point
    # ------------------------------------------------------------------
    def _posterior_params(
        self, p: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute posterior drift and vol given belief p.

        Parameters
        ----------
        p : (Np,) or scalar — belief (probability of regime 0)

        Returns
        -------
        mu_bar : (Np, d) or (d,)
        sigma_bar : (Np, d) or (d,)
        """
        p = np.atleast_1d(p)
        # belief = [p, 1-p] for 2 regimes
        beliefs = np.column_stack([p, 1 - p])  # (Np, 2)
        mu_bar = beliefs @ self.params.mu       # (Np, d)
        sigma_bar = beliefs @ self.params.sigma  # (Np, d)
        return mu_bar, sigma_bar

    # ------------------------------------------------------------------
    # Hamiltonian optimisation (single asset simplified for d=1 case)
    # ------------------------------------------------------------------
    def optimize_control(
        self,
        V_flat: NDArray[np.float64],
        V_x: NDArray[np.float64],
        V_xx: NDArray[np.float64],
        pi_prev: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Optimise the Hamiltonian over portfolio weights at each grid point.

        For small d (≤ 2), uses vectorised analytical solution when no
        transaction costs, and bounded scalar optimisation otherwise.

        Parameters
        ----------
        V_flat : (Nx*Np,) value function
        V_x : (Nx*Np,) first derivative ∂V/∂x
        V_xx : (Nx*Np,) second derivative ∂²V/∂x²
        pi_prev : (Nx*Np, d) or None — previous control (for TC)

        Returns
        -------
        pi_star : (Nx*Np, d) optimal portfolio weights
        H_star : (Nx*Np,) Hamiltonian at optimal control
        """
        N = len(V_flat)
        d = self.n_assets
        grid = self.grid

        # Posterior parameters at each belief — (Np, d)
        mu_bar, sigma_bar = self._posterior_params(grid.bg.p)

        # Broadcast to full grid (N,) by repeating per-wealth
        # Flat indexing: k = i_x * Np + i_p, so mu_k[k] = mu_bar[k % Np]
        p_indices = np.arange(N) % grid.Np
        mu_full = mu_bar[p_indices]         # (N, d)
        sig_full = sigma_bar[p_indices]     # (N, d)

        eps = self.params.transaction_cost
        theta = self.params.theta

        # Regularise V_xx to prevent sign flips at boundaries
        # For CRRA with gamma < 0, V_xx should be negative (concave V)
        vxx_reg = np.copy(V_xx)
        vxx_reg = np.where(np.abs(vxx_reg) < 1e-20, -1e-20, vxx_reg)
        # For gamma < 1 (our case), enforce concavity
        if self.gamma < 1:
            vxx_reg = np.minimum(vxx_reg, -1e-20)

        # Excess returns: (N, d)
        excess = mu_full - self.r
        vol_sq = sig_full ** 2  # (N, d) — diagonal covariance assumption

        # Robust control adjustment
        if theta > 0:
            safe_V = np.where(np.abs(V_flat) > 1e-20, V_flat, -1e-20)
            robust_adj = vol_sq * (V_x / (theta * safe_V))[:, None]
            excess = excess - robust_adj

        # Correct log-wealth FOC for optimal weights:
        # In log-wealth x = log(W), the HJB Hamiltonian is:
        #   H = [r + π·(μ-r) - ½π²σ²]·V_x + ½π²σ²·V_xx
        # FOC ∂H/∂π = 0 gives:
        #   π* = (μ-r)·V_x / [σ²·(V_x - V_xx)]
        # This recovers π* = (μ-r)/((1-γ)σ²) for CRRA V(x) = e^(γx)/γ.
        denom = V_x[:, None] - vxx_reg[:, None]  # (N, 1)
        # Regularise denominator (should be γV(1-γ) > 0 for γ < 1)
        denom = np.where(np.abs(denom) < 1e-20, np.sign(denom + 1e-30) * 1e-20, denom)
        pi_star = excess * V_x[:, None] / (vol_sq * denom)  # (N, d)

        # Clip to wide but reasonable bounds
        pi_star = np.clip(pi_star, -5.0, 5.0)

        # Apply transaction cost adjustment
        if eps > 0 and pi_prev is not None:
            tc_gradient = eps * np.sign(pi_star - pi_prev) * np.abs(V_x)[:, None]
            pi_star = pi_star - tc_gradient / (vxx_reg[:, None] * vol_sq + 1e-30)
            pi_star = np.clip(pi_star, -5.0, 5.0)

        # Evaluate explicit Hamiltonian: ADVECTION ONLY (drift · V_x)
        # The diffusion term (½ σ² V_xx) is handled implicitly by L_diff in time_loop
        port_vol_sq = np.sum((pi_star * sig_full) ** 2, axis=1)  # (N,)
        drift = self.r + np.sum(pi_star * (mu_full - self.r), axis=1) - 0.5 * port_vol_sq
        H_advection = drift * V_x

        # Subtract TC penalty
        if eps > 0 and pi_prev is not None:
            H_advection -= eps * np.sum(np.abs(pi_star - pi_prev), axis=1) * np.abs(V_x)

        return pi_star, H_advection

    # ------------------------------------------------------------------
    # No-trade region detection
    # ------------------------------------------------------------------
    def no_trade_region(
        self,
        pi_star: NDArray[np.float64],
        pi_merton: NDArray[np.float64],
        tol: float = 0.01,
    ) -> NDArray[np.bool_]:
        """Detect no-trade region where optimal control equals previous position.

        Parameters
        ----------
        pi_star : (Nx*Np, d) optimal portfolio
        pi_merton : (Nx*Np, d) Merton (frictionless) portfolio
        tol : float — tolerance for detecting no-trade

        Returns
        -------
        mask : (Nx*Np,) bool — True where in no-trade region
        """
        deviation = np.max(np.abs(pi_star - pi_merton), axis=1)
        return deviation > tol

    # ------------------------------------------------------------------
    # Belief dynamics coefficients (for the p-direction of the PDE)
    # ------------------------------------------------------------------
    def belief_drift(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Drift of the belief process (from Wonham filter).

        For 2 regimes with generator Q = [[-q01, q01], [q10, -q10]]:
            dp/dt term = q10(1-p) - q01*p = q10 - (q01+q10)*p

        Parameters
        ----------
        p : (...,) belief values

        Returns
        -------
        drift : (...,) array
        """
        Q = self.params.Q
        q01 = Q[0, 1]  # rate from regime 0 → 1
        q10 = Q[1, 0]  # rate from regime 1 → 0
        return q10 * (1 - p) - q01 * p

    def belief_diffusion(
        self, p: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Diffusion coefficient of the belief process.

        Squared diffusion: p²(1-p)² |h₀ - h₁|² where h_k is the
        signal-to-noise per regime.

        Parameters
        ----------
        p : (...,) belief values

        Returns
        -------
        diff_sq : (...,) array — squared diffusion coefficient
        """
        mu = self.params.mu     # (K, d)
        sigma = self.params.sigma  # (K, d)
        # h_k = mu_k / sigma_k^2 (simplified signal-to-noise)
        h = mu / (sigma ** 2 + 1e-30)
        h_diff_sq = np.sum((h[0] - h[1]) ** 2)  # scalar for 2 regimes
        return (p ** 2) * ((1 - p) ** 2) * h_diff_sq
