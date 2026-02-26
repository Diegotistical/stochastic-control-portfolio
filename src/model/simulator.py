"""
Monte Carlo simulator orchestrating dynamics, regime, and noise.

Produces tuples of (price_paths, regime_paths, observations) for
both synthetic experiments and filtered-data mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.common.config import MarketConfig, RegimeConfig
from src.model.dynamics import MultiAssetGBM
from src.model.noise import cholesky_factor, generate_increments
from src.model.regime import HiddenMarkovRegime


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation output."""

    prices: NDArray[np.float64]  # (n_paths, n_steps+1, n_assets)
    regimes: NDArray[np.int64]  # (n_paths, n_steps+1)
    log_returns: NDArray[np.float64]  # (n_paths, n_steps, n_assets)
    dt: float
    n_paths: int
    n_steps: int


class MonteCarloSimulator:
    """Orchestrates correlated GBM simulation with hidden-Markov regime switching.

    Parameters
    ----------
    market_cfg : MarketConfig
    regime_cfg : RegimeConfig
    """

    def __init__(self, market_cfg: MarketConfig, regime_cfg: RegimeConfig):
        self.market_cfg = market_cfg
        self.regime_cfg = regime_cfg

        self.gbm = MultiAssetGBM(
            mu=np.array(market_cfg.mu),
            sigma=np.array(market_cfg.sigma),
            correlation=np.array(market_cfg.correlation),
            risk_free_rate=market_cfg.risk_free_rate,
        )
        self.hmm = HiddenMarkovRegime(
            generator=np.array(regime_cfg.generator),
            initial_distribution=np.array(regime_cfg.initial_distribution),
        )
        self.chol = cholesky_factor(np.array(market_cfg.correlation))

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        S0: NDArray[np.float64] | None = None,
        seed: int | None = None,
        antithetic: bool = False,
    ) -> SimulationResult:
        """Run a full Monte Carlo simulation.

        Parameters
        ----------
        T : float
            Time horizon.
        n_steps : int
            Number of time steps.
        n_paths : int
            Number of Monte Carlo paths.
        S0 : (n_assets,) array, optional
            Initial asset prices.  Defaults to ones.
        seed : int, optional
            Random seed.
        antithetic : bool
            Use antithetic variates for variance reduction.

        Returns
        -------
        SimulationResult
        """
        dt = T / n_steps
        rng = np.random.default_rng(seed)

        if S0 is None:
            S0 = np.ones(self.market_cfg.n_assets)

        # Generate regime paths
        regimes = self.hmm.simulate(n_paths, n_steps, dt, rng)

        # Generate correlated Brownian increments
        dW = generate_increments(
            n_paths=n_paths,
            n_steps=n_steps,
            n_assets=self.market_cfg.n_assets,
            dt=dt,
            chol=self.chol,
            rng=rng,
            antithetic=antithetic,
        )

        # Simulate price paths
        prices = self.gbm.simulate_paths(S0, regimes, dW, dt)
        log_ret = MultiAssetGBM.log_returns(prices)

        return SimulationResult(
            prices=prices,
            regimes=regimes,
            log_returns=log_ret,
            dt=dt,
            n_paths=n_paths,
            n_steps=n_steps,
        )
