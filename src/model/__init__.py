"""Model sub-package: dynamics, regime, filtering, calibration, grid, operators, HJB."""

from src.model.dynamics import MultiAssetGBM
from src.model.regime import HiddenMarkovRegime
from src.model.noise import cholesky_factor, generate_increments
from src.model.simulator import MonteCarloSimulator, SimulationResult
from src.model.wonham_filter import WonhamFilter
from src.model.calibration import HMMCalibrator, CalibrationResult, download_data

__all__ = [
    "MultiAssetGBM",
    "HiddenMarkovRegime",
    "cholesky_factor",
    "generate_increments",
    "MonteCarloSimulator",
    "SimulationResult",
    "WonhamFilter",
    "HMMCalibrator",
    "CalibrationResult",
    "download_data",
]
