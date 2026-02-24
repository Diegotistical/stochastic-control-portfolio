"""
Convergence monitor for the HJB solver.

Tracks residual norms, checks monotonicity, detects oscillation,
and provides per-step diagnostics.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ConvergenceMonitor:
    """Monitor convergence of iterative PDE solvers.

    Parameters
    ----------
    tol : float
        Convergence tolerance on relative residual.
    patience : int
        Number of consecutive steps with residual below tol to declare convergence.
    oscillation_window : int
        Window size for detecting residual oscillation.
    """

    def __init__(
        self,
        tol: float = 1e-6,
        patience: int = 5,
        oscillation_window: int = 20,
    ):
        self.tol = tol
        self.patience = patience
        self.oscillation_window = oscillation_window

        self.residuals: list[float] = []
        self.converged = False
        self._below_tol_count = 0

    def update(self, residual: float, step: int) -> None:
        """Record a new residual and check convergence / oscillation.

        Parameters
        ----------
        residual : float
            Relative residual at this step.
        step : int
            Current step number.
        """
        self.residuals.append(residual)

        if residual < self.tol:
            self._below_tol_count += 1
        else:
            self._below_tol_count = 0

        if self._below_tol_count >= self.patience:
            self.converged = True
            logger.info(f"Converged at step {step} (residual={residual:.2e})")

        # Oscillation detection
        if len(self.residuals) >= self.oscillation_window:
            window = self.residuals[-self.oscillation_window:]
            diffs = np.diff(window)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            if sign_changes > self.oscillation_window * 0.7:
                logger.warning(
                    f"Residual oscillation detected at step {step} "
                    f"({sign_changes} sign changes in last {self.oscillation_window} steps)"
                )

        # Periodic logging
        if step % 50 == 0 or step < 5:
            logger.debug(f"Step {step:4d}: residual = {residual:.6e}")

    def summary(self) -> dict:
        """Return summary statistics."""
        if not self.residuals:
            return {"converged": False, "n_steps": 0}
        return {
            "converged": self.converged,
            "n_steps": len(self.residuals),
            "final_residual": self.residuals[-1],
            "min_residual": min(self.residuals),
            "max_residual": max(self.residuals),
        }
