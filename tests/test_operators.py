"""Unit tests for FD operators and spatial grid."""

import numpy as np
import pytest

from src.model.grid import WealthGrid, BeliefGrid, ProductGrid
from src.model.operators import FDOperators, _first_deriv_1d_central, _second_deriv_1d


class TestGrid:
    def test_wealth_grid_bounds(self):
        wg = WealthGrid(W_min=0.1, W_max=5.0, N=50)
        assert wg.W[0] >= 0.1
        assert wg.W[-1] <= 5.0
        assert len(wg.x) == 50

    def test_belief_grid_bounds(self):
        bg = BeliefGrid(N=30)
        assert bg.p[0] > 0
        assert bg.p[-1] < 1
        assert len(bg.p) == 30

    def test_product_grid_flat_round_trip(self):
        wg = WealthGrid(N=20)
        bg = BeliefGrid(N=10)
        grid = ProductGrid(wg, bg)

        arr = np.random.randn(20, 10)
        flat = grid.to_flat(arr)
        recovered = grid.to_2d(flat)
        np.testing.assert_array_equal(arr, recovered)

    def test_product_grid_total(self):
        wg = WealthGrid(N=25)
        bg = BeliefGrid(N=15)
        grid = ProductGrid(wg, bg)
        assert grid.total == 25 * 15


class TestFDOperators:
    def test_first_deriv_linear(self):
        """D1 of linear function f(x) = 2x + 1 should be ≈ 2."""
        x = np.linspace(0, 5, 50)
        f = 2 * x + 1
        D1 = _first_deriv_1d_central(x)
        df = D1 @ f
        np.testing.assert_allclose(df[1:-1], 2.0, atol=1e-10)

    def test_second_deriv_quadratic(self):
        """D2 of f(x) = x² should be ≈ 2."""
        x = np.linspace(0, 5, 100)
        f = x ** 2
        D2 = _second_deriv_1d(x)
        d2f = D2 @ f
        np.testing.assert_allclose(d2f[2:-2], 2.0, atol=0.1)

    def test_second_deriv_cubic(self):
        """D2 of f(x) = x³ should be ≈ 6x."""
        x = np.linspace(1, 5, 200)
        f = x ** 3
        D2 = _second_deriv_1d(x)
        d2f = D2 @ f
        expected = 6 * x
        np.testing.assert_allclose(d2f[5:-5], expected[5:-5], rtol=0.02)

    def test_operators_shape(self):
        """FDOperators produce correct matrix dimensions."""
        wg = WealthGrid(N=30)
        bg = BeliefGrid(N=20)
        grid = ProductGrid(wg, bg)
        ops = FDOperators(grid)

        N = grid.total
        assert ops.D1x.shape == (N, N)
        assert ops.D2x.shape == (N, N)
        assert ops.D1p.shape == (N, N)
