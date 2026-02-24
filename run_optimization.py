#!/usr/bin/env python3
"""
Deep Stochastic Control Portfolio Optimization — Main Entry Point

Usage:
    python run_optimization.py --config configs/default_config.yaml
    python run_optimization.py --config configs/default_config.yaml --solver hjb
    python run_optimization.py --config configs/default_config.yaml --solver both --backtest
    python run_optimization.py --config configs/default_config.yaml --sensitivity
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep Stochastic Control Portfolio Optimization"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--solver", type=str, choices=["hjb", "rl", "both"], default="hjb",
        help="Which solver to run",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run historical backtesting",
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Run sensitivity analysis",
    )
    parser.add_argument(
        "--robust", action="store_true",
        help="Run robust control comparison",
    )
    parser.add_argument(
        "--complexity", action="store_true",
        help="Run complexity analysis (PDE vs RL scaling)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    from src.common.config import load_config, set_global_seed

    logger.info(f"Loading config from {args.config}")
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Simulate synthetic data
    # ------------------------------------------------------------------
    from src.model.simulator import MonteCarloSimulator

    logger.info("Running Monte Carlo simulation...")
    simulator = MonteCarloSimulator(cfg.market, cfg.regime)
    sim_result = simulator.simulate(
        T=cfg.solver.T,
        n_steps=cfg.solver.n_time_steps,
        n_paths=1000,
        seed=cfg.seed,
    )
    logger.info(
        f"Simulated {sim_result.n_paths} paths, "
        f"{sim_result.n_steps} steps, dt={sim_result.dt:.6f}"
    )

    # ------------------------------------------------------------------
    # 3. Wonham filter
    # ------------------------------------------------------------------
    from src.model.wonham_filter import WonhamFilter

    logger.info("Running Wonham filter...")
    wf = WonhamFilter(
        Q=np.array(cfg.regime.generator),
        mu=np.array(cfg.market.mu),
        sigma=np.array(cfg.market.sigma),
        correlation=np.array(cfg.market.correlation),
    )
    beliefs = wf.filter(sim_result.log_returns, sim_result.dt)
    logger.info(f"Filter output shape: {beliefs.shape}")

    # Plot filter accuracy for first path
    from src.viz.static import plot_filter_accuracy
    plot_filter_accuracy(
        beliefs[0], sim_result.regimes[0],
        sim_result.dt, save_path=str(output_dir / "filter_accuracy.png"),
    )

    # ------------------------------------------------------------------
    # 4. HJB PDE Solver
    # ------------------------------------------------------------------
    if args.solver in ("hjb", "both"):
        from src.model.grid import WealthGrid, BeliefGrid, ProductGrid
        from src.model.hjb_model import HJBModel, HJBParams
        from src.solver.time_loop import HJBSolver
        from src.viz.static import plot_value_surface, plot_optimal_control, plot_convergence

        logger.info("Setting up HJB PDE solver...")

        wg = WealthGrid(
            W_min=cfg.solver.wealth_min,
            W_max=cfg.solver.wealth_max,
            N=cfg.solver.n_wealth_grid,
        )
        bg = BeliefGrid(N=cfg.solver.n_belief_grid)
        grid = ProductGrid(wg, bg)

        hjb_params = HJBParams(
            gamma=cfg.solver.gamma,
            r=cfg.market.risk_free_rate,
            mu=np.array(cfg.market.mu),
            sigma=np.array(cfg.market.sigma),
            correlation=np.array(cfg.market.correlation),
            Q=np.array(cfg.regime.generator),
            transaction_cost=cfg.market.transaction_cost,
            theta=cfg.robust.theta if cfg.robust.enabled else 0.0,
        )
        model = HJBModel(hjb_params, grid)
        solver = HJBSolver(
            model, grid,
            T=cfg.solver.T,
            n_steps=cfg.solver.n_time_steps,
            tol=cfg.solver.tol,
            adaptive_dt=cfg.solver.adaptive_dt,
        )

        logger.info("Solving HJB equation (backward in time)...")
        solution = solver.solve()
        logger.info(
            f"HJB solver: converged={solution.converged}, "
            f"final residual={solution.residuals[-1]:.2e}"
        )

        # Visualise
        V_t0 = grid.to_2d(solution.V[0])
        pi_t0 = solution.pi_star[0]
        if pi_t0.ndim == 2:
            pi_t0_2d = grid.to_2d(pi_t0[:, 0])
        else:
            pi_t0_2d = grid.to_2d(pi_t0)

        plot_value_surface(V_t0, wg.x, bg.p, save_path=str(output_dir / "value_surface.png"))
        plot_optimal_control(pi_t0_2d, wg.x, bg.p, save_path=str(output_dir / "optimal_control.png"))
        plot_convergence(solution.residuals, save_path=str(output_dir / "convergence.png"))

    # ------------------------------------------------------------------
    # 5. Deep RL Solver
    # ------------------------------------------------------------------
    if args.solver in ("rl", "both"):
        from src.solver.rl_env import VectorizedPortfolioEnv
        from src.solver.actor_critic import ActorCritic
        from src.solver.rl_trainer import PPOTrainer

        logger.info("Setting up Deep RL solver...")

        env = VectorizedPortfolioEnv(
            n_envs=cfg.rl.batch_size,
            mu=np.array(cfg.market.mu),
            sigma=np.array(cfg.market.sigma),
            correlation=np.array(cfg.market.correlation),
            Q=np.array(cfg.regime.generator),
            r=cfg.market.risk_free_rate,
            gamma=cfg.solver.gamma,
            T=cfg.solver.T,
            n_steps=cfg.rl.n_env_steps,
            transaction_cost=cfg.market.transaction_cost,
            n_assets=cfg.market.n_assets,
        )

        ac_model = ActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_sizes=cfg.rl.hidden_sizes,
        )

        trainer = PPOTrainer(
            env=env,
            model=ac_model,
            lr=cfg.rl.learning_rate,
            gamma=cfg.rl.gamma_discount,
            clip_epsilon=cfg.rl.clip_epsilon,
            entropy_coef=cfg.rl.entropy_coef,
            n_epochs=cfg.rl.n_epochs_per_update,
            checkpoint_dir=cfg.rl.checkpoint_dir,
        )

        logger.info(f"Training RL agent for {cfg.rl.n_episodes} episodes...")
        rl_metrics = trainer.train(n_episodes=cfg.rl.n_episodes)
        logger.info(
            f"RL training complete. Final mean wealth: "
            f"{rl_metrics.mean_wealth[-1]:.4f}"
        )

    # ------------------------------------------------------------------
    # 6. Robust control comparison
    # ------------------------------------------------------------------
    if args.robust or cfg.robust.compare_with_standard:
        from src.model.robust_control import RobustController
        from src.viz.static import plot_robust_comparison

        logger.info("Running robust control comparison...")
        rc = RobustController(
            theta=cfg.robust.theta,
            gamma=cfg.solver.gamma,
            r=cfg.market.risk_free_rate,
        )
        mu_avg = np.array(cfg.market.mu).mean(axis=0)
        sigma_avg = np.array(cfg.market.sigma).mean(axis=0)

        pi_std, pi_rob = rc.compare_allocations(mu_avg, sigma_avg)
        logger.info(f"Standard allocation: {pi_std}")
        logger.info(f"Robust allocation:   {pi_rob}")
        logger.info(f"Exposure reduction:  {1 - np.abs(pi_rob)/np.abs(pi_std)}")

        plot_robust_comparison(
            pi_std, pi_rob,
            labels=[f"Asset {i+1}" for i in range(len(pi_std))],
            save_path=str(output_dir / "robust_comparison.png"),
        )

    # ------------------------------------------------------------------
    # 7. Complexity analysis
    # ------------------------------------------------------------------
    if args.complexity:
        from src.analysis.complexity import analyze_scaling, format_complexity_table
        from src.viz.static import plot_complexity_scaling

        logger.info("Running complexity analysis...")
        cx = analyze_scaling()
        print(format_complexity_table(cx))

        plot_complexity_scaling(
            cx.dimensions, cx.pde_flops, cx.rl_flops_per_step,
            save_path=str(output_dir / "complexity_scaling.png"),
        )

    # ------------------------------------------------------------------
    # 8. Benchmarks
    # ------------------------------------------------------------------
    from src.benchmarks.merton import MertonStrategy

    mu_avg = np.array(cfg.market.mu).mean(axis=0)
    sigma_avg = np.array(cfg.market.sigma).mean(axis=0)

    merton = MertonStrategy(mu_avg, sigma_avg, cfg.market.risk_free_rate, cfg.solver.gamma)
    merton_sol = merton.solve(W0=1.0, T=cfg.solver.T)
    logger.info(f"Merton optimal weights: {merton_sol.pi_star}")
    logger.info(f"Merton CE wealth: {merton_sol.certainty_equivalent:.4f}")

    # ------------------------------------------------------------------
    # 9. Backtesting (optional)
    # ------------------------------------------------------------------
    if args.backtest and cfg.backtest.enabled:
        from src.model.calibration import download_data, HMMCalibrator
        from src.backtest.engine import BacktestEngine
        from src.backtest.metrics import compute_metrics
        from src.benchmarks.mean_variance import MeanVarianceStrategy
        from src.benchmarks.risk_parity import RiskParityStrategy
        from src.viz.static import plot_benchmark_comparison

        logger.info("Downloading historical data...")
        train_returns = download_data(
            cfg.backtest.tickers, cfg.backtest.train_start, cfg.backtest.train_end
        )
        test_returns = download_data(
            cfg.backtest.tickers, cfg.backtest.test_start, cfg.backtest.test_end
        )

        logger.info("Calibrating HMM from training data...")
        calibrator = HMMCalibrator(n_regimes=cfg.regime.n_regimes)
        cal_result = calibrator.fit(train_returns)
        logger.info(f"Calibrated μ: {cal_result.mu}")
        logger.info(f"Calibrated σ: {cal_result.sigma}")

        # Define strategies as callables
        def merton_strategy(hist, cur_w):
            ms = MertonStrategy(cal_result.mu.mean(axis=0),
                                np.sqrt(np.diag(cal_result.covariances.mean(axis=0))),
                                cfg.market.risk_free_rate, cfg.solver.gamma)
            return np.clip(ms.optimal_weights(), 0, 1)

        def mv_strategy(hist, cur_w):
            mvs = MeanVarianceStrategy.from_returns(hist, cfg.market.risk_free_rate)
            return mvs.optimal_weights(allow_short=False)

        def rp_strategy(hist, cur_w):
            rps = RiskParityStrategy.from_returns(hist)
            return rps.optimal_weights()

        engine = BacktestEngine(
            test_returns,
            initial_wealth=cfg.backtest.initial_wealth,
            transaction_cost=cfg.market.transaction_cost,
            rebalance_freq=cfg.backtest.rebalance_freq,
        )

        strategies = {
            "Merton": merton_strategy,
            "Mean-Variance": mv_strategy,
            "Risk-Parity": rp_strategy,
        }

        results = engine.run_multiple(strategies)
        comparison_data = {}
        for name, bt_result in results.items():
            metrics = compute_metrics(
                bt_result.returns, bt_result.wealth_path,
                bt_result.turnover, bt_result.transaction_costs,
            )
            comparison_data[name] = {
                "wealth": bt_result.wealth_path,
                "metrics": {
                    "annualised_return": metrics.annualised_return,
                    "annualised_volatility": metrics.annualised_volatility,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown": metrics.max_drawdown,
                    "sortino_ratio": metrics.sortino_ratio,
                },
            }
            logger.info(f"{name}: Sharpe={metrics.sharpe_ratio:.2f}, MaxDD={metrics.max_drawdown:.2%}")

        plot_benchmark_comparison(
            comparison_data, save_path=str(output_dir / "backtest_comparison.png")
        )

    logger.info(f"All outputs saved to {output_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
