"""
Finite-difference operators for the HJB PDE on the product grid.

Constructs sparse (CSR) matrices for:
  - First derivative (central + upwind) in x = log(W)
  - First derivative in p (belief)
  - Second derivative in x
  - Neumann boundary conditions
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from src.model.grid import ProductGrid


def _first_deriv_1d_central(
    x: NDArray[np.float64],
) -> sp.csr_matrix:
    """Central difference first derivative on a (possibly non-uniform) 1D grid.

    Uses second-order centred stencil:
        f'(x_i) ≈ (f_{i+1} - f_{i-1}) / (x_{i+1} - x_{i-1})

    Boundary rows are set to one-sided (forward/backward) differences.

    Parameters
    ----------
    x : (N,) array — grid points

    Returns
    -------
    D1 : (N, N) sparse CSR matrix
    """
    N = len(x)
    data, row, col = [], [], []

    # Interior: central
    for i in range(1, N - 1):
        h_total = x[i + 1] - x[i - 1]
        data.extend([-1.0 / h_total, 1.0 / h_total])
        row.extend([i, i])
        col.extend([i - 1, i + 1])

    # Forward difference at i=0
    h0 = x[1] - x[0]
    data.extend([-1.0 / h0, 1.0 / h0])
    row.extend([0, 0])
    col.extend([0, 1])

    # Backward difference at i=N-1
    hN = x[-1] - x[-2]
    data.extend([-1.0 / hN, 1.0 / hN])
    row.extend([N - 1, N - 1])
    col.extend([N - 2, N - 1])

    return sp.csr_matrix((data, (row, col)), shape=(N, N))


def _second_deriv_1d(
    x: NDArray[np.float64],
) -> sp.csr_matrix:
    """Second derivative on a (possibly non-uniform) 1D grid.

    Uses the standard three-point stencil for non-uniform grids:
        f''(x_i) ≈ 2 / (h_i + h_{i+1}) * [f_{i+1}/h_{i+1} - f_i*(1/h_i + 1/h_{i+1}) + f_{i-1}/h_i]

    Boundary rows are zero (Neumann conditions imposed separately).

    Parameters
    ----------
    x : (N,) array

    Returns
    -------
    D2 : (N, N) sparse CSR matrix
    """
    N = len(x)
    data, row, col = [], [], []

    for i in range(1, N - 1):
        h_l = x[i] - x[i - 1]
        h_r = x[i + 1] - x[i]
        denom = 0.5 * (h_l + h_r)

        data.extend(
            [
                1.0 / (h_l * denom),
                -1.0 / (h_l * denom) - 1.0 / (h_r * denom),
                1.0 / (h_r * denom),
            ]
        )
        row.extend([i, i, i])
        col.extend([i - 1, i, i + 1])

    return sp.csr_matrix((data, (row, col)), shape=(N, N))


class FDOperators:
    """Collection of finite-difference operators on the product grid.

    Attributes
    ----------
    D1x : (Nx*Np, Nx*Np) sparse — ∂/∂x (central)
    D2x : (Nx*Np, Nx*Np) sparse — ∂²/∂x²
    D1p : (Nx*Np, Nx*Np) sparse — ∂/∂p (central)
    """

    def __init__(self, grid: ProductGrid):
        self.grid = grid
        Nx = grid.Nx
        Np = grid.Np

        # 1D operators
        D1x_1d = _first_deriv_1d_central(grid.wg.x)
        D2x_1d = _second_deriv_1d(grid.wg.x)
        D1p_1d = _first_deriv_1d_central(grid.bg.p)

        # Kronecker product to lift to 2D product grid
        Ip = sp.eye(Np, format="csr")
        Ix = sp.eye(Nx, format="csr")

        self.D1x = sp.kron(D1x_1d, Ip, format="csr")
        self.D2x = sp.kron(D2x_1d, Ip, format="csr")
        self.D1p = sp.kron(Ix, D1p_1d, format="csr")

    def apply_neumann_bc(
        self,
        V_flat: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Enforce Neumann (zero-flux) boundary conditions.

        Sets ∂V/∂x = 0 at x_min and x_max for all p.
        Sets ∂V/∂p = 0 at p = 0 and p = 1 for all x.

        Parameters
        ----------
        V_flat : (Nx*Np,) array

        Returns
        -------
        V_flat : (Nx*Np,) array with boundaries adjusted.
        """
        V = self.grid.to_2d(V_flat).copy()
        Nx, Np = self.grid.Nx, self.grid.Np

        # Neumann in x: V[0, :] = V[1, :], V[-1, :] = V[-2, :]
        V[0, :] = V[1, :]
        V[Nx - 1, :] = V[Nx - 2, :]

        # Neumann in p: V[:, 0] = V[:, 1], V[:, -1] = V[:, -2]
        V[:, 0] = V[:, 1]
        V[:, Np - 1] = V[:, Np - 2]

        return self.grid.to_flat(V)

    def upwind_D1x(
        self,
        velocity: NDArray[np.float64],
    ) -> sp.csr_matrix:
        """Build upwind first-derivative operator in x based on velocity sign.

        Uses forward difference where velocity > 0, backward where < 0.

        Parameters
        ----------
        velocity : (Nx*Np,) array
            Advection velocity at each grid point.

        Returns
        -------
        D1x_upwind : (Nx*Np, Nx*Np) sparse matrix
        """
        grid = self.grid
        Nx, Np = grid.Nx, grid.Np
        x = grid.wg.x

        data, row, col = [], [], []

        for i in range(Nx):
            for j in range(Np):
                flat = grid.flat_index(i, j)
                v = velocity[flat]

                if i == 0:
                    # Forward difference at left boundary
                    h = x[1] - x[0]
                    data.extend([-1.0 / h, 1.0 / h])
                    row.extend([flat, flat])
                    col.extend([flat, grid.flat_index(1, j)])
                elif i == Nx - 1:
                    # Backward difference at right boundary
                    h = x[-1] - x[-2]
                    data.extend([-1.0 / h, 1.0 / h])
                    row.extend([flat, flat])
                    col.extend([grid.flat_index(Nx - 2, j), flat])
                elif v >= 0:
                    # Backward (upwind for positive velocity)
                    h = x[i] - x[i - 1]
                    data.extend([-1.0 / h, 1.0 / h])
                    row.extend([flat, flat])
                    col.extend([grid.flat_index(i - 1, j), flat])
                else:
                    # Forward (upwind for negative velocity)
                    h = x[i + 1] - x[i]
                    data.extend([-1.0 / h, 1.0 / h])
                    row.extend([flat, flat])
                    col.extend([flat, grid.flat_index(i + 1, j)])

        return sp.csr_matrix((data, (row, col)), shape=(Nx * Np, Nx * Np))
