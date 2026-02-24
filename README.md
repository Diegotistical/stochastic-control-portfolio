# Deep Stochastic Control for Multi-Asset Portfolio Optimization

A comprehensive Python framework implementing **optimal portfolio allocation** under transaction costs, partial information (hidden Markov regimes), and model ambiguity, using both **HJB PDE solvers** and **Deep Reinforcement Learning**.

## Key Features

| Feature | Description |
|---|---|
| **HJB PDE Solver** | Finite-difference IMEX scheme for 2-asset + risk-free, with sinh-stretched grids |
| **Deep RL Solver** | PPO with Dirichlet actor for arbitrary $d$ assets (scalability demo) |
| **Regime Switching** | Hidden Markov Model with EM calibration from historical data |
| **Wonham Filter** | Real-time posterior regime estimation from observed returns |
| **Robust Control** | Hansen–Sargent ambiguity aversion with worst-case drift distortion |
| **Transaction Costs** | Proportional costs generating no-trade regions |
| **Benchmarks** | Merton closed-form, mean-variance, risk-parity strategies |
| **Backtesting** | Walk-forward engine with comprehensive performance metrics |

## Mathematical Framework

The investor solves:

$$V(W, p, t) = \sup_\pi \inf_h \mathbb{E}\left[\frac{W_T^\gamma}{\gamma} + \int_t^T \frac{\theta}{2}|h_s|^2 ds\right]$$

- **CRRA utility** with parameter $\gamma < 1$
- **Regime-switching GBM** with generator $Q$
- **Partial information** via Wonham filter on belief state $p$
- **Ambiguity aversion** via Hansen–Sargent penalty $\theta > 0$

See `docs/derivations.md` for complete mathematical derivations.

## Repository Structure

```
├── configs/
│   └── default_config.yaml        # All tuneable parameters
├── docs/
│   └── derivations.md             # Mathematical derivations & references
├── src/
│   ├── common/config.py           # Dataclass config + YAML I/O
│   ├── exceptions/                # Custom exception hierarchy
│   ├── model/
│   │   ├── dynamics.py            # Multi-asset GBM
│   │   ├── regime.py              # CTMC regime model
│   │   ├── noise.py               # Correlated Wiener increments
│   │   ├── simulator.py           # Monte Carlo orchestrator
│   │   ├── wonham_filter.py       # Wonham filter
│   │   ├── calibration.py         # HMM calibration (EM / Baum–Welch)
│   │   ├── grid.py                # Sinh-stretched wealth + belief grids
│   │   ├── operators.py           # Finite-difference operators (Kronecker)
│   │   ├── hjb_model.py           # HJB model (Hamiltonian, TC, robust)
│   │   └── robust_control.py      # Hansen–Sargent robust controller
│   ├── solver/
│   │   ├── time_loop.py           # IMEX HJB backward solver
│   │   ├── monitor.py             # Convergence tracking
│   │   ├── rl_env.py              # Gym-style portfolio environment
│   │   ├── actor_critic.py        # Dirichlet actor + value critic
│   │   └── rl_trainer.py          # PPO training loop with GAE
│   ├── benchmarks/
│   │   ├── merton.py              # Merton closed-form (1969/1971)
│   │   ├── mean_variance.py       # Markowitz MVO
│   │   └── risk_parity.py         # Equal risk contribution
│   ├── backtest/
│   │   ├── engine.py              # Walk-forward backtester
│   │   └── metrics.py             # Sharpe, Sortino, MaxDD, VaR, CVaR
│   ├── analysis/
│   │   ├── sensitivity.py         # 1D/2D parameter sweeps
│   │   ├── robustness.py          # Model misspecification tests
│   │   └── complexity.py          # PDE vs RL scaling analysis
│   └── viz/
│       └── static.py              # Publication-quality matplotlib plots
├── tests/
│   ├── test_dynamics.py           # GBM log-return moments
│   ├── test_regime.py             # Stationary dist, transition matrix
│   ├── test_wonham.py             # Simplex preservation, convergence
│   ├── test_operators.py          # FD accuracy on polynomials
│   ├── test_hjb_merton.py         # Critical: HJB recovers Merton
│   ├── test_benchmarks.py         # Analytical weight verification
│   ├── test_rl_env.py             # Env contract (obs, rewards, termination)
│   └── test_metrics.py            # Sharpe, drawdown edge cases
├── run_optimization.py            # Main CLI entry point
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run HJB solver
python run_optimization.py --config configs/default_config.yaml --solver hjb

# Run both solvers + robust comparison
python run_optimization.py --solver both --robust

# Full pipeline with backtesting
python run_optimization.py --solver both --backtest --robust --complexity

# Run tests
pytest tests/ -v
```

## Design Highlights

### Curse of Dimensionality
The HJB PDE solver is intentionally restricted to **2 risky assets** + 1 risk-free. The Deep RL solver supports **arbitrary `d`** assets. The complexity analysis demonstrates that PDE grid sizes scale as $O(N^{d+1})$ while RL parameters scale as $O(d)$.

### Robust Control
Hansen–Sargent multiplier preferences introduce a worst-case drift distortion $h^* = -\sigma \pi V_W / \theta$, which **shrinks** portfolio exposure relative to the standard Merton solution. This protects against model misspecification.

### Numerical Verification
The critical test verifies that the HJB solver **recovers the Merton closed-form** under $\varepsilon = 0$ (no transaction costs) and single regime (no switching), providing a gold-standard regression test.

## Configuration

All parameters are specified in `configs/default_config.yaml`:
- **Market**: drifts, volatilities, correlation, transaction costs
- **Regime**: generator matrix, initial distribution
- **Solver**: grid sizes, time horizon, CRRA parameter, tolerance
- **Robust**: ambiguity parameter $\theta$
- **RL**: network architecture, PPO hyperparameters
- **Backtest**: tickers, date ranges, rebalance frequency

## References

1. Merton, R.C. (1969). *Lifetime Portfolio Selection*. Review of Economics and Statistics.
2. Hansen, L.P. & Sargent, T.J. (2001). *Robust Control and Model Uncertainty*. AER.
3. Wonham, W.M. (1965). *Some Applications of Stochastic Differential Equations to Optimal Nonlinear Filtering*. SIAM J. Control.
4. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
5. Davis, M.H.A. & Norman, A.R. (1990). *Portfolio Selection with Transaction Costs*. MOR.
