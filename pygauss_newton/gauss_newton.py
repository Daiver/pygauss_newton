import time
from typing import Callable, Union
import numpy as np


class Settings:
    def __init__(self, n_max_iterations=50, dumping_constant=0.0, verbose=True):
        self.n_max_iterations = n_max_iterations
        self.dumping_constant = dumping_constant
        self.verbose = verbose


class OptimizationState:
    """
    Should be implemented later
    """
    def __init__(self):
        pass


def gauss_newton(
        residuals_func: Callable,
        jacobian_func: Callable,
        x0: Union[np.ndarray],
        settings: Settings = None,
        update_functor: Callable = None):
    optimization_start_time = time.time()
    if settings is None:
        settings = Settings()
    if not (type(x0) is np.ndarray):
        x0 = np.array(x0, dtype=np.float32)
    assert x0.dtype in [np.float, np.float32, np.float64]

    optimization_state = OptimizationState()

    x = x0.copy()
    n_variables = len(x)
    eye = np.eye(n_variables)
    for iter_ind in range(settings.n_max_iterations):
        residuals_val = residuals_func(x)
        jacobian_val = jacobian_func(x)
        assert residuals_val.ndim == 1
        n_residuals = len(residuals_val)

        assert jacobian_val.ndim == 2
        assert jacobian_val.shape == (n_residuals, n_variables)

        gradient_val = jacobian_val.T @ residuals_val
        loss_val = 0.5 * residuals_val.T @ residuals_val

        hessian_val = jacobian_val.T @ jacobian_val
        hessian_val += settings.dumping_constant * eye

        step_val = -np.linalg.solve(hessian_val, gradient_val)

        if settings.verbose:
            print(
                f"{iter_ind + 1}/{settings.n_max_iterations}. "
                f"Loss = {loss_val}, "
                f"|grad| = {np.linalg.norm(gradient_val)} "
                f"|step| = {np.linalg.norm(step_val)} "
            )
        x += step_val
        if update_functor is not None:
            if update_functor(x, optimization_state) is False:
                break
    if settings.verbose:
        print(f"Optimization elapsed: {time.time() - optimization_start_time}")

    return x, optimization_state
