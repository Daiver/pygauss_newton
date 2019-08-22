from typing import Callable
import numpy as np


class Settings:
    def __init__(self, n_max_iterations=50, dumping_constant=0.0, verbose=True):
        self.n_max_iterations = n_max_iterations
        self.dumping_constant = dumping_constant
        self.verbose = verbose


def gauss_newton(
        residuals_func: Callable,
        jacobian_func: Callable,
        x0: np.ndarray,
        settings: Settings = None):

    if settings is None:
        settings = Settings()
    x = x0.copy()
    n_variables = len(x)
    eye = np.eye(n_variables)
    for iter_ind in range(settings.n_max_iterations):
        residuals_val = residuals_func(x)
        jacobian_val = jacobian_func(x)

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
    return x
