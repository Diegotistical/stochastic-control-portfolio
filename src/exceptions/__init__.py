"""Custom exception hierarchy for the stochastic control framework."""


class StochasticControlError(Exception):
    """Base exception for all framework errors."""


class ConfigValidationError(StochasticControlError):
    """Raised when configuration parameters are invalid or inconsistent."""


class NumericalInstabilityError(StochasticControlError):
    """Raised when a numerical scheme produces NaN, Inf, or violates stability bounds."""

    def __init__(self, message: str, step: int | None = None, residual: float | None = None):
        self.step = step
        self.residual = residual
        detail = message
        if step is not None:
            detail += f" (step={step})"
        if residual is not None:
            detail += f" (residual={residual:.2e})"
        super().__init__(detail)


class SolverConvergenceError(StochasticControlError):
    """Raised when the HJB solver fails to converge within the specified tolerance."""

    def __init__(self, message: str, iterations: int | None = None, final_residual: float | None = None):
        self.iterations = iterations
        self.final_residual = final_residual
        detail = message
        if iterations is not None:
            detail += f" (iterations={iterations})"
        if final_residual is not None:
            detail += f" (residual={final_residual:.2e})"
        super().__init__(detail)


class CalibrationError(StochasticControlError):
    """Raised when HMM calibration (EM/MLE) fails to converge or produces degenerate parameters."""


class FilterDegeneracyError(StochasticControlError):
    """Raised when the Wonham filter belief state escapes the probability simplex."""
