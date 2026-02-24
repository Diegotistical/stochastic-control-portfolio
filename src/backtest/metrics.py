"""
Performance metrics for portfolio backtesting.

All metrics are annualised unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PerformanceMetrics:
    """Container for strategy performance statistics."""
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    total_turnover: float
    avg_turnover_per_rebalance: float
    total_transaction_costs: float
    skewness: float
    kurtosis: float
    var_95: float           # 5% Value at Risk (daily)
    cvar_95: float          # 5% Conditional VaR (daily)
    final_wealth: float


def compute_metrics(
    returns: NDArray[np.float64],
    wealth: NDArray[np.float64],
    turnover: NDArray[np.float64] | None = None,
    tc_paid: NDArray[np.float64] | None = None,
    periods_per_year: float = 252.0,
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics.

    Parameters
    ----------
    returns : (T,) daily portfolio returns
    wealth : (T+1,) wealth path
    turnover : (T,) daily turnover (optional)
    tc_paid : (T,) daily transaction costs paid (optional)
    periods_per_year : float

    Returns
    -------
    PerformanceMetrics
    """
    # Filter out zeros (pre-lookback period)
    active = returns != 0
    if active.sum() < 2:
        # Not enough data
        return PerformanceMetrics(
            annualised_return=0, annualised_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            calmar_ratio=0, total_turnover=0, avg_turnover_per_rebalance=0,
            total_transaction_costs=0, skewness=0, kurtosis=0,
            var_95=0, cvar_95=0, final_wealth=wealth[-1],
        )

    r = returns[active]

    ann_ret = np.mean(r) * periods_per_year
    ann_vol = np.std(r, ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

    # Sortino (downside deviation)
    downside = r[r < 0]
    down_vol = np.std(downside, ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else 1e-10
    sortino = ann_ret / down_vol if down_vol > 1e-10 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(wealth)
    dd = (peak - wealth) / (peak + 1e-30)
    max_dd = dd.max()

    calmar = ann_ret / max_dd if max_dd > 1e-10 else 0.0

    # Turnover
    total_to = turnover.sum() if turnover is not None else 0.0
    n_rebalances = (turnover > 0).sum() if turnover is not None else 1
    avg_to = total_to / max(n_rebalances, 1)

    # Transaction costs
    total_tc = tc_paid.sum() if tc_paid is not None else 0.0

    # Higher moments
    skew = float(np.mean(((r - r.mean()) / (r.std() + 1e-30)) ** 3))
    kurt = float(np.mean(((r - r.mean()) / (r.std() + 1e-30)) ** 4) - 3)

    # VaR and CVaR
    var_95 = float(-np.percentile(r, 5))
    cvar_95 = float(-np.mean(r[r <= -var_95])) if (r <= -var_95).sum() > 0 else var_95

    return PerformanceMetrics(
        annualised_return=float(ann_ret),
        annualised_volatility=float(ann_vol),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=float(max_dd),
        calmar_ratio=float(calmar),
        total_turnover=float(total_to),
        avg_turnover_per_rebalance=float(avg_to),
        total_transaction_costs=float(total_tc),
        skewness=skew,
        kurtosis=kurt,
        var_95=var_95,
        cvar_95=cvar_95,
        final_wealth=float(wealth[-1]),
    )
