"""
Configuration system for the stochastic control portfolio framework.

All parameters are grouped into dataclasses and can be loaded from YAML.
Seed management ensures full reproducibility.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.exceptions import ConfigValidationError


# ---------------------------------------------------------------------------
# Dataclass definitions
# ---------------------------------------------------------------------------

@dataclass
class MarketConfig:
    """Parameters for multi-asset GBM dynamics."""
    n_assets: int = 2
    risk_free_rate: float = 0.03
    # Per-regime drift vectors: shape (n_regimes, n_assets)
    mu: list[list[float]] = field(default_factory=lambda: [
        [0.08, 0.12],   # bull regime
        [0.02, 0.04],   # bear regime
    ])
    # Per-regime volatility vectors: shape (n_regimes, n_assets)
    sigma: list[list[float]] = field(default_factory=lambda: [
        [0.15, 0.20],
        [0.25, 0.35],
    ])
    # Correlation matrix (constant across regimes for simplicity)
    correlation: list[list[float]] = field(default_factory=lambda: [
        [1.0, 0.5],
        [0.5, 1.0],
    ])
    # Transaction cost rate (proportional)
    transaction_cost: float = 0.001

    def validate(self) -> None:
        if self.n_assets < 1:
            raise ConfigValidationError("n_assets must be >= 1")
        if self.risk_free_rate < 0:
            raise ConfigValidationError("risk_free_rate must be non-negative")
        if self.transaction_cost < 0:
            raise ConfigValidationError("transaction_cost must be non-negative")
        mu_arr = np.array(self.mu)
        sigma_arr = np.array(self.sigma)
        if mu_arr.ndim != 2 or mu_arr.shape[1] != self.n_assets:
            raise ConfigValidationError(
                f"mu must be (n_regimes, {self.n_assets}), got {mu_arr.shape}"
            )
        if sigma_arr.ndim != 2 or sigma_arr.shape[1] != self.n_assets:
            raise ConfigValidationError(
                f"sigma must be (n_regimes, {self.n_assets}), got {sigma_arr.shape}"
            )
        if sigma_arr.shape[0] != mu_arr.shape[0]:
            raise ConfigValidationError("mu and sigma must have same number of regimes")
        corr = np.array(self.correlation)
        if corr.shape != (self.n_assets, self.n_assets):
            raise ConfigValidationError(
                f"correlation must be ({self.n_assets}, {self.n_assets})"
            )
        if not np.allclose(corr, corr.T):
            raise ConfigValidationError("correlation matrix must be symmetric")
        eigvals = np.linalg.eigvalsh(corr)
        if np.any(eigvals < -1e-10):
            raise ConfigValidationError("correlation matrix must be positive semi-definite")


@dataclass
class RegimeConfig:
    """Parameters for the hidden Markov regime-switching model."""
    n_regimes: int = 2
    # Generator matrix Q: shape (n_regimes, n_regimes), rows sum to 0
    generator: list[list[float]] = field(default_factory=lambda: [
        [-0.5, 0.5],
        [1.0, -1.0],
    ])
    # Initial regime probability distribution
    initial_distribution: list[float] = field(default_factory=lambda: [0.5, 0.5])

    def validate(self) -> None:
        Q = np.array(self.generator)
        if Q.shape != (self.n_regimes, self.n_regimes):
            raise ConfigValidationError(
                f"generator must be ({self.n_regimes}, {self.n_regimes})"
            )
        row_sums = Q.sum(axis=1)
        if not np.allclose(row_sums, 0.0, atol=1e-10):
            raise ConfigValidationError(
                f"generator rows must sum to 0, got sums={row_sums}"
            )
        if np.any(Q[np.eye(self.n_regimes, dtype=bool)] > 0):
            raise ConfigValidationError("diagonal of generator must be non-positive")
        p0 = np.array(self.initial_distribution)
        if len(p0) != self.n_regimes:
            raise ConfigValidationError("initial_distribution length must match n_regimes")
        if not np.isclose(p0.sum(), 1.0):
            raise ConfigValidationError("initial_distribution must sum to 1")


@dataclass
class SolverConfig:
    """Parameters for the HJB PDE solver."""
    # Time horizon
    T: float = 1.0
    # Number of time steps
    n_time_steps: int = 200
    # Grid sizes
    n_wealth_grid: int = 100
    n_belief_grid: int = 50
    # Wealth domain [W_min, W_max]
    wealth_min: float = 0.01
    wealth_max: float = 10.0
    # CRRA risk aversion parameter (gamma < 0 for risk-averse, != 0)
    gamma: float = -2.0
    # Convergence tolerance
    tol: float = 1e-6
    max_iterations: int = 1000
    # Whether to use adaptive time stepping
    adaptive_dt: bool = True
    # CFL safety factor (< 1)
    cfl_factor: float = 0.8

    def validate(self) -> None:
        if self.T <= 0:
            raise ConfigValidationError("T must be positive")
        if self.gamma == 0:
            raise ConfigValidationError("gamma must be nonzero for CRRA utility")
        if self.gamma >= 1:
            raise ConfigValidationError("gamma must be < 1 for CRRA (typically negative)")
        if self.wealth_min <= 0:
            raise ConfigValidationError("wealth_min must be positive")
        if self.wealth_max <= self.wealth_min:
            raise ConfigValidationError("wealth_max must exceed wealth_min")


@dataclass
class RobustConfig:
    """Parameters for Hansen-Sargent ambiguity aversion."""
    enabled: bool = True
    # Ambiguity aversion parameter (theta > 0; larger = less ambiguity averse)
    theta: float = 0.1
    # Whether to solve both robust and standard for comparison
    compare_with_standard: bool = True

    def validate(self) -> None:
        if self.enabled and self.theta <= 0:
            raise ConfigValidationError("theta must be positive when robust control is enabled")


@dataclass
class RLConfig:
    """Parameters for the Deep RL solver."""
    enabled: bool = True
    # Network architecture
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128, 64])
    # Training
    n_episodes: int = 5000
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma_discount: float = 0.99
    entropy_coef: float = 0.01
    clip_epsilon: float = 0.2  # PPO clip
    n_epochs_per_update: int = 4
    # Environment
    n_env_steps: int = 252  # daily steps for 1 year
    # Number of assets (can be > 2 to demonstrate scalability)
    n_assets_rl: int = 2
    checkpoint_dir: str = "checkpoints/"

    def validate(self) -> None:
        if self.n_episodes < 1:
            raise ConfigValidationError("n_episodes must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigValidationError("learning_rate must be positive")


@dataclass
class BacktestConfig:
    """Parameters for backtesting."""
    enabled: bool = True
    # Tickers for historical data
    tickers: list[str] = field(default_factory=lambda: ["XLF", "XLK"])
    # Train / test split dates
    train_start: str = "2005-01-01"
    train_end: str = "2015-12-31"
    test_start: str = "2016-01-01"
    test_end: str = "2023-12-31"
    # Rebalance frequency in trading days
    rebalance_freq: int = 21  # monthly
    # Initial wealth
    initial_wealth: float = 1.0

    def validate(self) -> None:
        if len(self.tickers) < 1:
            raise ConfigValidationError("At least one ticker required")
        if self.initial_wealth <= 0:
            raise ConfigValidationError("initial_wealth must be positive")


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    market: MarketConfig = field(default_factory=MarketConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    robust: RobustConfig = field(default_factory=RobustConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    seed: int = 42
    output_dir: str = "outputs/"

    def validate(self) -> None:
        """Validate all sub-configs and cross-config consistency."""
        self.market.validate()
        self.regime.validate()
        self.solver.validate()
        self.robust.validate()
        self.rl.validate()
        self.backtest.validate()
        # Cross-config checks
        n_regimes_market = np.array(self.market.mu).shape[0]
        if n_regimes_market != self.regime.n_regimes:
            raise ConfigValidationError(
                f"market.mu implies {n_regimes_market} regimes but "
                f"regime.n_regimes={self.regime.n_regimes}"
            )


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def _nested_dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively instantiate nested dataclasses from dict."""
    from dataclasses import fields as dc_fields
    fieldtypes = {f.name: f.type for f in dc_fields(cls)}
    kwargs = {}
    for k, v in data.items():
        if k in fieldtypes and isinstance(v, dict):
            # Try to resolve the type annotation to a dataclass
            ft = fieldtypes[k]
            if isinstance(ft, str):
                ft = globals().get(ft, None)
            if ft is not None and hasattr(ft, "__dataclass_fields__"):
                kwargs[k] = _nested_dataclass_from_dict(ft, v)
            else:
                kwargs[k] = v
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str | Path) -> PipelineConfig:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    cfg = _nested_dataclass_from_dict(PipelineConfig, raw)
    cfg.validate()
    return cfg


def save_config(cfg: PipelineConfig, path: str | Path) -> None:
    """Save configuration to a YAML file."""
    from dataclasses import asdict
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def set_global_seed(seed: int) -> None:
    """Set seeds for numpy, torch, and python random for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
