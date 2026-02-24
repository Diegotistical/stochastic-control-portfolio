"""Analysis sub-package: sensitivity, robustness, complexity."""

from src.analysis.sensitivity import (
    run_sensitivity_1d,
    run_sensitivity_2d,
    SensitivityResult,
)
from src.analysis.robustness import (
    test_drift_misspecification,
    test_volatility_misspecification,
    test_regime_misspecification,
    RobustnessResult,
)
from src.analysis.complexity import analyze_scaling, ComplexityResult

__all__ = [
    "run_sensitivity_1d",
    "run_sensitivity_2d",
    "SensitivityResult",
    "test_drift_misspecification",
    "test_volatility_misspecification",
    "test_regime_misspecification",
    "RobustnessResult",
    "analyze_scaling",
    "ComplexityResult",
]
