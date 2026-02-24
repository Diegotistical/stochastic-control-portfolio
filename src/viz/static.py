"""
Publication-quality visualisation for the stochastic control framework.

All plotting functions produce matplotlib figures suitable for
inclusion in a research report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray


# --- Style configuration ---
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_value_surface(
    V: NDArray[np.float64],
    x_grid: NDArray[np.float64],
    p_grid: NDArray[np.float64],
    title: str = "Value Function V(W, p)",
    save_path: str | None = None,
) -> plt.Figure:
    """3D surface plot of the value function.

    Parameters
    ----------
    V : (Nx, Np) array
    x_grid : (Nx,) log-wealth grid
    p_grid : (Np,) belief grid
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    X, P = np.meshgrid(x_grid, p_grid, indexing="ij")
    W = np.exp(X)

    surf = ax.plot_surface(W, P, V, cmap="viridis", alpha=0.9, edgecolor="none")
    ax.set_xlabel("Wealth W")
    ax.set_ylabel("Belief p (P[bull])")
    ax.set_zlabel("V(W, p)")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, label="Value")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_optimal_control(
    pi: NDArray[np.float64],
    x_grid: NDArray[np.float64],
    p_grid: NDArray[np.float64],
    asset_idx: int = 0,
    title: str = "Optimal Portfolio Weight π*(W, p)",
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of optimal portfolio weight.

    Parameters
    ----------
    pi : (Nx, Np) or (Nx, Np, d) array
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    if pi.ndim == 3:
        pi = pi[:, :, asset_idx]

    W = np.exp(x_grid)
    im = ax.pcolormesh(p_grid, W, pi, cmap="RdBu_r", shading="auto")
    ax.set_xlabel("Belief p (P[bull])")
    ax.set_ylabel("Wealth W")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=f"π*₍{asset_idx+1}₎")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_no_trade_region(
    pi_star: NDArray[np.float64],
    pi_merton: NDArray[np.float64],
    x_grid: NDArray[np.float64],
    p_grid: NDArray[np.float64],
    tc_values: list[float] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Visualise no-trade regions for different transaction cost levels.

    Parameters
    ----------
    pi_star : (Nx, Np) optimal portfolio weight (with TC)
    pi_merton : (Nx, Np) Merton optimal (no TC)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    W = np.exp(x_grid)

    # Show deviation from Merton
    deviation = pi_star - pi_merton
    im = ax.pcolormesh(p_grid, W, deviation, cmap="coolwarm", shading="auto",
                       norm=Normalize(vmin=-0.3, vmax=0.3))
    ax.set_xlabel("Belief p (P[bull])")
    ax.set_ylabel("Wealth W")
    ax.set_title("No-Trade Region (deviation from Merton)")
    fig.colorbar(im, ax=ax, label="π* − π_Merton")

    # Overlay no-trade boundary as contour
    ax.contour(p_grid, W, np.abs(deviation), levels=[0.01], colors="black",
               linewidths=2, linestyles="--")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_benchmark_comparison(
    results: dict[str, dict[str, Any]],
    save_path: str | None = None,
) -> plt.Figure:
    """Equity curves and performance table for multiple strategies.

    Parameters
    ----------
    results : dict mapping strategy_name → {"wealth": array, "dates": list, "metrics": dict}
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curves
    ax1 = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    for i, (name, data) in enumerate(results.items()):
        wealth = data.get("wealth", np.ones(10))
        ax1.plot(wealth, label=name, color=colors[i], alpha=0.9)

    ax1.set_ylabel("Wealth")
    ax1.set_title("Strategy Comparison — Equity Curves")
    ax1.legend(loc="upper left")
    ax1.set_yscale("log")

    # Performance table
    ax2 = axes[1]
    ax2.axis("off")

    if results:
        headers = ["Strategy", "Ann. Ret.", "Ann. Vol.", "Sharpe", "Max DD", "Sortino"]
        table_data = []
        for name, data in results.items():
            m = data.get("metrics", {})
            table_data.append([
                name,
                f"{m.get('annualised_return', 0):.2%}",
                f"{m.get('annualised_volatility', 0):.2%}",
                f"{m.get('sharpe_ratio', 0):.2f}",
                f"{m.get('max_drawdown', 0):.2%}",
                f"{m.get('sortino_ratio', 0):.2f}",
            ])

        table = ax2.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_sensitivity_heatmap(
    values: NDArray[np.float64],
    x_labels: NDArray[np.float64],
    y_labels: NDArray[np.float64],
    x_name: str = "Parameter 1",
    y_name: str = "Parameter 2",
    metric_name: str = "Metric",
    save_path: str | None = None,
) -> plt.Figure:
    """2D sensitivity heatmap."""
    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.pcolormesh(x_labels, y_labels, values.T, cmap="YlOrRd", shading="auto")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"Sensitivity Analysis: {metric_name}")
    fig.colorbar(im, ax=ax, label=metric_name)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_filter_accuracy(
    beliefs: NDArray[np.float64],
    true_regimes: NDArray[np.int64],
    dt: float,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot Wonham filter belief vs true regime for a single path."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    T = len(beliefs)
    times = np.arange(T) * dt

    ax1.plot(times, beliefs[:, 0], label="P(bull)", color="steelblue", alpha=0.8)
    if beliefs.shape[1] > 1:
        ax1.plot(times, beliefs[:, 1], label="P(bear)", color="firebrick", alpha=0.8)
    ax1.set_ylabel("Posterior Probability")
    ax1.set_title("Wonham Filter: Belief State")
    ax1.legend()

    # True regime (stepped)
    T_reg = min(len(true_regimes), T)
    ax2.fill_between(times[:T_reg], true_regimes[:T_reg], alpha=0.3, color="gray", step="post")
    ax2.step(times[:T_reg], true_regimes[:T_reg], color="black", where="post", linewidth=1)
    ax2.set_ylabel("True Regime")
    ax2.set_xlabel("Time")
    ax2.set_title("True Regime Path")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Bull", "Bear"])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_convergence(
    residuals: list[float],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot solver convergence (residual vs iteration)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(residuals, color="darkblue", alpha=0.8)
    ax.set_xlabel("Time Step (backward)")
    ax.set_ylabel("Relative Residual")
    ax.set_title("HJB Solver Convergence")
    ax.axhline(y=1e-6, color="red", linestyle="--", alpha=0.5, label="Tolerance")
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_complexity_scaling(
    dimensions: NDArray,
    pde_flops: NDArray,
    rl_flops: NDArray,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot PDE vs RL computational scaling."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(dimensions, pde_flops, "o-", color="firebrick", label="PDE (Finite Differences)", markersize=8)
    ax.semilogy(dimensions, rl_flops, "s-", color="steelblue", label="Deep RL (Forward Pass)", markersize=8)

    ax.set_xlabel("Number of Assets (d)")
    ax.set_ylabel("FLOPs per Step")
    ax.set_title("Curse of Dimensionality: PDE vs Deep RL Scaling")
    ax.legend()

    # Annotate the crossover
    ax.fill_between(dimensions, pde_flops, rl_flops, alpha=0.1, color="gray")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_robust_comparison(
    pi_standard: NDArray[np.float64],
    pi_robust: NDArray[np.float64],
    labels: list[str],
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart comparing standard vs robust allocations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, pi_standard, width, label="Standard", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, pi_robust, width, label="Robust (Ambiguity-Averse)", color="firebrick", alpha=0.8)

    ax.set_xlabel("Asset")
    ax.set_ylabel("Optimal Weight π*")
    ax.set_title("Effect of Ambiguity Aversion on Portfolio Allocation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
