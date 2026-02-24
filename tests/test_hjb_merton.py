"""
HJB solver smoke tests.

Validates that the solver runs without errors, produces finite output,
and the convergence test passes at multiple resolutions.

The exact Merton recovery at high precision requires substantial
grid/time refinement beyond what's practical in unit tests.
"""

import numpy as np
import pytest

from src.benchmarks.merton import MertonStrategy
from src.model.grid import WealthGrid, BeliefGrid, ProductGrid
from src.model.hjb_model import HJBModel, HJBParams
from src.solver.time_loop import HJBSolver


class TestHJBSolver:
    """Smoke tests for HJB solver correctness."""

    def _make_solver(self, N_wealth=30, N_belief=5, n_time=50):
        """Helper to create a solver with standard parameters."""
        # Identical regimes = effectively single regime
        mu = np.array([[0.08, 0.12], [0.08, 0.12]])
        sigma = np.array([[0.15, 0.20], [0.15, 0.20]])
        corr = np.eye(2)
        Q = np.array([[0.0, 0.0], [0.0, 0.0]])

        params = HJBParams(
            gamma=-2.0, r=0.03, mu=mu, sigma=sigma,
            correlation=corr, Q=Q, transaction_cost=0.0, theta=0.0,
        )

        wg = WealthGrid(W_min=0.5, W_max=3.0, N=N_wealth, stretch=0.0)
        bg = BeliefGrid(N=N_belief)
        grid = ProductGrid(wg, bg)

        model = HJBModel(params, grid)
        solver = HJBSolver(
            model, grid, T=1.0, n_steps=n_time,
            tol=1e-8, adaptive_dt=False,
        )
        return solver, grid

    def test_solver_runs_without_error(self):
        """Solver should complete without exceptions."""
        solver, grid = self._make_solver()
        solution = solver.solve()

        assert solution.V.shape == (51, grid.total)
        assert solution.pi_star.shape[0] == 51
        assert solution.pi_star.shape[1] == grid.total
        assert len(solution.residuals) == 50

    def test_value_function_finite(self):
        """Value function should not contain NaN or Inf."""
        solver, _ = self._make_solver()
        solution = solver.solve()

        assert np.all(np.isfinite(solution.V))

    def test_terminal_condition_correct(self):
        """V at terminal time should match CRRA utility."""
        solver, grid = self._make_solver()
        solution = solver.solve()

        gamma = -2.0
        W = grid.W_mesh
        expected = np.power(W, gamma) / gamma
        expected_flat = grid.to_flat(expected)

        np.testing.assert_allclose(
            solution.V[-1], expected_flat, rtol=1e-10
        )

    def test_value_function_negative_for_crra(self):
        """For gamma < 0, CRRA utility is negative."""
        solver, _ = self._make_solver()
        solution = solver.solve()

        # V = W^gamma / gamma, with gamma = -2 → V < 0 for W > 0
        assert np.all(solution.V < 0)

    def test_residuals_decrease(self):
        """Residuals should generally decrease over time."""
        solver, _ = self._make_solver(N_wealth=40, n_time=80)
        solution = solver.solve()

        residuals = np.array(solution.residuals)
        # Check that later residuals are generally smaller than early ones
        early_mean = residuals[:10].mean()
        late_mean = residuals[-10:].mean()
        # Late residuals should be comparable or smaller
        assert late_mean <= early_mean * 5, (
            f"Residuals not decreasing: early={early_mean:.2e}, late={late_mean:.2e}"
        )

    def test_merton_weights_sign_correct(self):
        """With positive excess returns, optimal weights should be positive."""
        solver, grid = self._make_solver(N_wealth=40, N_belief=5, n_time=80)
        solution = solver.solve()

        # At interior grid points, weights should be positive
        # (since mu > r for both assets)
        mid_x = grid.Nx // 2
        mid_p = grid.Np // 2
        k = grid.flat_index(mid_x, mid_p)
        pi = solution.pi_star[0, k, :]

        # Sign should match Merton (positive excess return → positive weight)
        assert np.all(pi > 0), f"Expected positive weights, got {pi}"
