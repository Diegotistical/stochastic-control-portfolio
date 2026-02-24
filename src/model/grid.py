"""
Spatial grid construction for the HJB PDE solver.

WealthGrid: non-uniform (sinh-stretched) 1D grid in log-wealth.
BeliefGrid: uniform grid on [0,1] for 2-regime belief coordinate.
ProductGrid: tensor product (log_W, p) with indexing helpers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class WealthGrid:
    """Non-uniform grid in log-wealth space using sinh stretching.

    The stretching concentrates grid points near a focal log-wealth
    value (typically log(W0)), improving resolution where V(W) varies
    most rapidly.

    Parameters
    ----------
    W_min, W_max : float
        Wealth bounds (in natural units, > 0).
    N : int
        Number of grid points.
    stretch : float
        Stretching factor (higher = more concentration near centre).
    """

    def __init__(
        self,
        W_min: float = 0.01,
        W_max: float = 10.0,
        N: int = 100,
        stretch: float = 3.0,
    ):
        self.W_min = W_min
        self.W_max = W_max
        self.N = N
        self.stretch = stretch

        # Build grid in log-wealth
        x_min = np.log(W_min)
        x_max = np.log(W_max)

        if stretch > 0:
            # Sinh-stretched grid: ξ uniform on [-1, 1], x = centre + scale * sinh(stretch * ξ)
            xi = np.linspace(-1, 1, N)
            centre = 0.5 * (x_min + x_max)
            scale = (x_max - x_min) / (2 * np.sinh(stretch))
            self.x = centre + scale * np.sinh(stretch * xi)
        else:
            self.x = np.linspace(x_min, x_max, N)

        self.W = np.exp(self.x)
        self.dx = np.diff(self.x)  # (N-1,) non-uniform spacing

    @property
    def h(self) -> NDArray[np.float64]:
        """Average spacing for convergence analysis."""
        return self.dx.mean()


class BeliefGrid:
    """Uniform grid on [0, 1] for the 2-regime belief coordinate.

    p represents P(regime = 0 | observations), so p ∈ [0, 1].

    Parameters
    ----------
    N : int
        Number of grid points.
    eps : float
        Offset from boundary to avoid degeneracy.
    """

    def __init__(self, N: int = 50, eps: float = 1e-4):
        self.N = N
        self.p = np.linspace(eps, 1 - eps, N)
        self.dp = self.p[1] - self.p[0]


class ProductGrid:
    """Tensor product grid (log_W, p) with 2D ↔ 1D indexing.

    Parameters
    ----------
    wealth_grid : WealthGrid
    belief_grid : BeliefGrid
    """

    def __init__(self, wealth_grid: WealthGrid, belief_grid: BeliefGrid):
        self.wg = wealth_grid
        self.bg = belief_grid
        self.Nx = wealth_grid.N
        self.Np = belief_grid.N
        self.total = self.Nx * self.Np

        # 2D meshgrid arrays (Nx, Np)
        self.X, self.P = np.meshgrid(wealth_grid.x, belief_grid.p, indexing="ij")
        self.W_mesh = np.exp(self.X)

    def to_flat(self, arr_2d: NDArray[np.float64]) -> NDArray[np.float64]:
        """Flatten (Nx, Np) → (Nx*Np,) in row-major order."""
        return arr_2d.ravel(order="C")

    def to_2d(self, arr_flat: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reshape (Nx*Np,) → (Nx, Np)."""
        return arr_flat.reshape(self.Nx, self.Np, order="C")

    def flat_index(self, i_x: int, i_p: int) -> int:
        """Get flat index from 2D indices."""
        return i_x * self.Np + i_p
