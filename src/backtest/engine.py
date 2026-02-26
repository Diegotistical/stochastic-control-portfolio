"""
Walk-forward backtesting engine.

Applies a strategy at each rebalance date, tracks portfolio value
after transaction costs, and records the full path for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Strategy(Protocol):
    """Protocol for strategy callable."""

    def __call__(
        self,
        returns_history: NDArray[np.float64],
        current_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return target weights given return history and current position."""
        ...


@dataclass
class BacktestResult:
    """Container for backtesting output."""

    dates: list
    wealth_path: NDArray[np.float64]
    weights_history: NDArray[np.float64]
    returns: NDArray[np.float64]
    turnover: NDArray[np.float64]
    transaction_costs: NDArray[np.float64]
    strategy_name: str


class BacktestEngine:
    """Walk-forward backtesting engine with transaction costs.

    Parameters
    ----------
    returns : DataFrame
        Historical log-returns with DatetimeIndex.
    initial_wealth : float
    transaction_cost : float
        Proportional cost rate.
    rebalance_freq : int
        Rebalance every N trading days.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        initial_wealth: float = 1.0,
        transaction_cost: float = 0.001,
        rebalance_freq: int = 21,
    ):
        self.returns = returns
        self.initial_wealth = initial_wealth
        self.tc = transaction_cost
        self.rebalance_freq = rebalance_freq
        self.n_assets = returns.shape[1]

    def run(
        self,
        strategy_fn: Callable,
        strategy_name: str = "strategy",
        lookback: int = 252,
    ) -> BacktestResult:
        """Run backtest for a single strategy.

        Parameters
        ----------
        strategy_fn : callable
            (returns_history, current_weights) → target_weights
        strategy_name : str
        lookback : int
            Number of days of return history to pass to strategy.

        Returns
        -------
        BacktestResult
        """
        ret_values = self.returns.values
        dates = self.returns.index.tolist()
        T = len(dates)

        wealth = np.empty(T + 1)
        wealth[0] = self.initial_wealth

        weights = np.zeros((T + 1, self.n_assets))
        turnover = np.zeros(T)
        tc_paid = np.zeros(T)
        daily_returns = np.zeros(T)

        current_weights = np.zeros(self.n_assets)

        for t in range(T):
            # Rebalance?
            if t % self.rebalance_freq == 0 and t >= lookback:
                history = ret_values[max(0, t - lookback) : t]
                target_weights = strategy_fn(history, current_weights)
                target_weights = np.clip(target_weights, 0, None)
                if target_weights.sum() > 1:
                    target_weights = target_weights / target_weights.sum()

                # Transaction costs
                trade = np.abs(target_weights - current_weights)
                tc_t = self.tc * trade.sum() * wealth[t]
                wealth[t] -= tc_t

                turnover[t] = trade.sum()
                tc_paid[t] = tc_t
                current_weights = target_weights.copy()

            weights[t + 1] = current_weights

            # Daily return
            asset_rets = np.exp(ret_values[t]) - 1
            rf_weight = 1 - current_weights.sum()
            port_ret = rf_weight * 0 + np.dot(current_weights, asset_rets)
            daily_returns[t] = port_ret

            wealth[t + 1] = wealth[t] * (1 + port_ret)

            # Drift weights due to price changes
            if wealth[t + 1] > 0:
                new_asset_values = current_weights * (1 + asset_rets)
                total = rf_weight * 1 + new_asset_values.sum()
                if total > 0:
                    current_weights = new_asset_values / total

        return BacktestResult(
            dates=dates,
            wealth_path=wealth,
            weights_history=weights,
            returns=daily_returns,
            turnover=turnover,
            transaction_costs=tc_paid,
            strategy_name=strategy_name,
        )

    def run_multiple(
        self,
        strategies: dict[str, Callable],
        lookback: int = 252,
    ) -> dict[str, BacktestResult]:
        """Run backtest for multiple strategies.

        Parameters
        ----------
        strategies : dict mapping name → strategy callable
        lookback : int

        Returns
        -------
        dict mapping name → BacktestResult
        """
        results = {}
        for name, fn in strategies.items():
            results[name] = self.run(fn, strategy_name=name, lookback=lookback)
        return results
