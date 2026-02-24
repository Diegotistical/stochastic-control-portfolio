"""
IMEX time-stepping solver for the HJB PDE.

Backward-in-time scheme:
  - Implicit: diffusion terms (linear, sparse solve)
  - Explicit: nonlinear Hamiltonian optimisation

Stores the optimal control surface π*(W, p, t) and value function V(W, p, t).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from src.exceptions import NumericalInstabilityError, SolverConvergenceError
from src.model.grid import ProductGrid
from src.model.hjb_model import HJBModel, HJBParams
from src.model.operators import FDOperators
from src.solver.monitor import ConvergenceMonitor


@dataclass
class HJBSolution:
    """Container for the HJB solver output."""
    V: NDArray[np.float64]              # (n_time, Nx*Np) value function at each time
    pi_star: NDArray[np.float64]        # (n_time, Nx*Np, d) optimal control
    times: NDArray[np.float64]          # (n_time,) time points
    grid: ProductGrid
    converged: bool
    residuals: list[float]


class HJBSolver:
    """Solve the HJB PDE via backward IMEX time-stepping.

    Parameters
    ----------
    model : HJBModel
    grid : ProductGrid
    T : float
        Time horizon.
    n_steps : int
        Number of time steps.
    tol : float
        Convergence tolerance.
    adaptive_dt : bool
        Use adaptive CFL-based time stepping.
    cfl_factor : float
        Safety factor for CFL condition (< 1).
    """

    def __init__(
        self,
        model: HJBModel,
        grid: ProductGrid,
        T: float = 1.0,
        n_steps: int = 200,
        tol: float = 1e-6,
        adaptive_dt: bool = True,
        cfl_factor: float = 0.8,
    ):
        self.model = model
        self.grid = grid
        self.T = T
        self.n_steps = n_steps
        self.tol = tol
        self.adaptive_dt = adaptive_dt
        self.cfl_factor = cfl_factor
        self.ops = FDOperators(grid)
        self.monitor = ConvergenceMonitor(tol=tol)

    def solve(self) -> HJBSolution:
        """Run the backward-in-time HJB solver.

        Returns
        -------
        HJBSolution
        """
        grid = self.grid
        model = self.model
        N = grid.total
        d = model.n_assets

        dt = self.T / self.n_steps
        times = np.linspace(self.T, 0, self.n_steps + 1)

        # Storage
        V_history = np.empty((self.n_steps + 1, N))
        pi_history = np.empty((self.n_steps + 1, N, d))

        # Terminal condition
        V = model.terminal_condition()
        V_history[-1] = V
        pi_history[-1] = np.zeros((N, d))

        residuals = []
        pi_prev = None

        for step in range(self.n_steps):
            t_idx = self.n_steps - step  # index counting backward

            # Compute derivatives
            V_x = self.ops.D1x @ V
            V_xx = self.ops.D2x @ V
            V_p = self.ops.D1p @ V

            # Explicit step: optimise Hamiltonian
            pi_star, H_explicit = model.optimize_control(V, V_x, V_xx, pi_prev)

            # Belief dynamics contribution
            p_flat = grid.to_flat(grid.P)
            belief_drift = model.belief_drift(p_flat)
            belief_diff = model.belief_diffusion(p_flat)

            # Explicit source term
            V_pp = self.ops.D1p @ V_p  # ∂²V/∂p² (approximate)
            source = H_explicit + belief_drift * V_p + 0.5 * belief_diff * V_pp

            # Adaptive dt based on CFL
            if self.adaptive_dt:
                dt_cfl = self._compute_cfl_dt(V, pi_star)
                dt_step = min(dt, dt_cfl)
            else:
                dt_step = dt

            # Implicit diffusion matrix: (I - dt * L_diff) V^{n+1} = V^n + dt * source
            # L_diff captures only the second-order x terms
            # For stability, we treat the diffusion implicitly
            sigma_bar = np.zeros(N)
            for i_p in range(grid.Np):
                _, sig = model._posterior_params(grid.bg.p[i_p:i_p + 1])
                sig = sig[0]  # (d,)
                for i_x in range(grid.Nx):
                    k = grid.flat_index(i_x, i_p)
                    pi = pi_star[k]
                    sigma_bar[k] = np.sqrt(np.sum((pi * sig) ** 2))

            # Build diagonal diffusion coefficient
            diff_coeff = 0.5 * sigma_bar ** 2
            L_diff = sp.diags(diff_coeff) @ self.ops.D2x

            # Implicit system: (I - dt * L_diff) V_new = V + dt * source
            A_implicit = sp.eye(N) - dt_step * L_diff
            rhs = V + dt_step * source

            # Apply boundary conditions
            rhs = self.ops.apply_neumann_bc(rhs)

            # Solve
            try:
                V_new = spla.spsolve(A_implicit.tocsc(), rhs)
            except Exception as e:
                raise NumericalInstabilityError(
                    f"Sparse solve failed: {e}", step=step
                )

            # Stability check
            if np.any(np.isnan(V_new)) or np.any(np.isinf(V_new)):
                raise NumericalInstabilityError(
                    "NaN or Inf detected in value function", step=step
                )

            # Apply boundary conditions to solution
            V_new = self.ops.apply_neumann_bc(V_new)

            # Compute residual
            residual = np.max(np.abs(V_new - V)) / (np.max(np.abs(V)) + 1e-30)
            residuals.append(residual)
            self.monitor.update(residual, step)

            # Store
            V_history[t_idx - 1] = V_new
            pi_history[t_idx - 1] = pi_star

            V = V_new
            pi_prev = pi_star

        return HJBSolution(
            V=V_history,
            pi_star=pi_history,
            times=times,
            grid=grid,
            converged=self.monitor.converged,
            residuals=residuals,
        )

    def _compute_cfl_dt(
        self,
        V: NDArray[np.float64],
        pi_star: NDArray[np.float64],
    ) -> float:
        """Compute CFL-limited time step.

        CFL condition for advection-diffusion:
            dt ≤ min(dx² / (2D), dx / |a|)

        where D is diffusion coefficient and a is advection velocity.
        """
        grid = self.grid
        dx_min = grid.wg.dx.min()
        dp_min = grid.bg.dp

        # Rough estimate of max diffusion
        sigma_max = np.max(self.model.params.sigma)
        pi_max = np.max(np.abs(pi_star))
        D_max = 0.5 * (pi_max * sigma_max) ** 2

        # Advection speed estimate
        mu_max = np.max(np.abs(self.model.params.mu))
        a_max = self.model.r + pi_max * mu_max

        dt_diff = dx_min ** 2 / (2 * D_max + 1e-30)
        dt_adv = dx_min / (a_max + 1e-30)
        dt_belief = dp_min ** 2 / (2 * np.max(np.abs(self.model.params.Q)) + 1e-30)

        return self.cfl_factor * min(dt_diff, dt_adv, dt_belief)
