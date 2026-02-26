"""
Hansen–Sargent robust control (ambiguity aversion).

Implements the worst-case drift distortion under KL penalty,
yielding a minimax HJB where the investor hedges against
model misspecification.

Robust HJB:
    V_t + sup_π inf_h { H(π, V) + h·σ·π·V_x + θ·|h|²/2 } = 0

The inner minimisation over h yields the worst-case distortion:
    h* = -(σ·π·V_x) / θ

Substituting back gives the effective drift adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RobustControlResult:
    """Container for robust control analysis."""

    pi_standard: NDArray[np.float64]  # standard optimal control
    pi_robust: NDArray[np.float64]  # robust optimal control
    h_worst: NDArray[np.float64]  # worst-case distortion
    value_standard: NDArray[np.float64]  # standard value function
    value_robust: NDArray[np.float64]  # robust value function
    exposure_reduction: NDArray[np.float64]  # |π_robust| / |π_standard|


class RobustController:
    """Compute robust optimal control under Hansen–Sargent preferences.

    The investor solves:
        max_π min_Q̃  E^{Q̃}[U(W_T)] + θ · KL(Q̃ || Q)

    where θ > 0 controls the level of ambiguity aversion.
    Smaller θ = more ambiguity averse (larger distortion).

    Parameters
    ----------
    theta : float
        Ambiguity aversion parameter (> 0).
    gamma : float
        CRRA risk aversion parameter.
    r : float
        Risk-free rate.
    """

    def __init__(self, theta: float, gamma: float, r: float):
        if theta <= 0:
            raise ValueError("theta must be positive")
        self.theta = theta
        self.gamma = gamma
        self.r = r

    def worst_case_distortion(
        self,
        pi: NDArray[np.float64],
        sigma: NDArray[np.float64],
        V_x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the worst-case drift distortion.

        h* = -(σ·π·V_x) / θ

        Parameters
        ----------
        pi : (..., d) portfolio weights
        sigma : (..., d) volatilities
        V_x : (...,) ∂V/∂x

        Returns
        -------
        h : (..., d) worst-case distortion vector
        """
        # h*_i = -(sigma_i * pi_i * V_x) / theta
        return -(sigma * pi * V_x[..., None]) / self.theta

    def robust_excess_return(
        self,
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        pi: NDArray[np.float64],
        V_x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute effective excess return under worst-case measure.

        μ̃ − r = (μ − r) + σ·h* = (μ − r) − σ²·π·V_x / θ

        Parameters
        ----------
        mu : (..., d) drift
        sigma : (..., d) volatility
        pi : (..., d) portfolio weights
        V_x : (...,) value function derivative

        Returns
        -------
        excess_robust : (..., d) effective excess return
        """
        h = self.worst_case_distortion(pi, sigma, V_x)
        return (mu - self.r) + sigma * h

    def robust_merton_ratio(
        self,
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        V_x: NDArray[np.float64],
        V_xx: NDArray[np.float64],
        V: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute robust Merton portfolio weight.

        For CRRA utility V(W) = W^γ/γ, the robust optimal weight is:

            π*_robust = (μ − r) / ((1−γ)σ² + σ²V_x/(θV))

        This shows the shrinkage of allocation under ambiguity.

        Parameters
        ----------
        mu : (..., d) drift
        sigma : (..., d) volatility
        V_x : (...,) first derivative
        V_xx : (...,) second derivative
        V : (...,) value function

        Returns
        -------
        pi_robust : (..., d) robust optimal weights
        """
        excess = mu - self.r
        vol_sq = sigma**2

        # Standard denominator: -V_xx/V_x * vol_sq = (1-gamma) * vol_sq for CRRA
        denom_standard = -(V_xx / (V_x + 1e-30))[..., None] * vol_sq

        # Ambiguity correction term
        ambiguity_term = vol_sq * (V_x / (self.theta * V + 1e-30))[..., None]

        pi_robust = excess / (denom_standard + ambiguity_term + 1e-30)
        return pi_robust

    def compare_allocations(
        self,
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compare standard vs robust Merton allocations analytically.

        For CRRA utility with power γ:
            π_standard = (μ−r) / ((1−γ)σ²)
            π_robust   = (μ−r) / ((1−γ)σ² + σ²/(θ·(γ−1)))

        Parameters
        ----------
        mu : (d,) drift
        sigma : (d,) volatility

        Returns
        -------
        pi_standard, pi_robust : (d,) arrays
        """
        excess = mu - self.r
        vol_sq = sigma**2
        gamma = self.gamma

        pi_standard = excess / ((1 - gamma) * vol_sq)

        # For CRRA V(W) = W^γ/γ: V_x/V = γ/W * γ/(W^γ/γ) simplified
        # The robust correction adds σ²/(θ(1-γ)) to the denominator
        correction = vol_sq / (self.theta * (1 - gamma))
        pi_robust = excess / ((1 - gamma) * vol_sq + correction)

        return pi_standard, pi_robust

    def kl_divergence_penalty(
        self,
        h: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute KL divergence penalty: θ/2 · |h|².

        Parameters
        ----------
        h : (..., d) distortion vector

        Returns
        -------
        penalty : (...,) scalar penalty per grid point
        """
        return 0.5 * self.theta * np.sum(h**2, axis=-1)
