import time
from typing import Callable, Union
import numpy as np

from .utils import time_fn
from .stopping_reason import StoppingReason


class Settings:
    def __init__(self,
                 n_max_iterations=50,
                 dumping_constant=0.0,
                 loss_stop_threshold=0.0,
                 grad_norm_stop_threshold=0.0,
                 step_norm_stop_threshold=0.0,
                 verbose=True):
        self.n_max_iterations = n_max_iterations
        self.dumping_constant = dumping_constant

        self.loss_stop_threshold = loss_stop_threshold
        self.grad_norm_stop_threshold = grad_norm_stop_threshold
        self.step_norm_stop_threshold = step_norm_stop_threshold

        self.verbose = verbose


class OptimizationState:
    def __init__(self):
        self.iter_ind = None
        self.variables_val = None
        self.residuals_val = None
        self.jacobian_val = None
        self.gradient_val = None
        self.gradient_norm = None
        self.loss_val = None
        self.hessian_val = None
        self.step_val = None
        self.step_norm = None
        self.stopping_reason = StoppingReason.NotStopped


def gauss_newton(
        residuals_func: Callable,
        jacobian_func: Callable,
        x0: Union[np.ndarray],
        settings: Settings = None,
        update_functor: Callable = None):
    start_time_optimization = time.time()
    if settings is None:
        settings = Settings()
    if not (type(x0) is np.ndarray):
        x0 = np.array(x0, dtype=np.float32)
    assert x0.dtype in [np.float, np.float32, np.float64]

    state = OptimizationState()

    state.variables_val = x0.copy()
    n_variables = len(state.variables_val)
    eye = np.eye(n_variables)

    for iter_ind in range(settings.n_max_iterations):
        state.iter_ind = iter_ind
        state.residuals_val, elapsed_residuals = time_fn(residuals_func, state.variables_val)
        state.jacobian_val, elapsed_jacobian = time_fn(jacobian_func, state.variables_val)

        assert state.residuals_val.ndim == 1
        n_residuals = len(state.residuals_val)

        assert state.jacobian_val.ndim == 2
        assert state.jacobian_val.shape == (n_residuals, n_variables)

        state.gradient_val = state.jacobian_val.T @ state.residuals_val
        state.gradient_norm = np.linalg.norm(state.gradient_val)
        state.loss_val = 0.5 * state.residuals_val.T @ state.residuals_val

        state.hessian_val = state.jacobian_val.T @ state.jacobian_val
        state.hessian_val += settings.dumping_constant * eye

        state.step_val = -np.linalg.solve(state.hessian_val, state.gradient_val)
        state.step_norm = np.linalg.norm(state.step_val)

        elapsed_upd = 0
        if update_functor is not None:
            functor_result, elapsed_upd = time_fn(update_functor, state.variables_val, state)
            if functor_result is False:
                state.stopping_reason = StoppingReason.ByCallback
                break
        if settings.verbose:
            print(
                f"{iter_ind + 1}/{settings.n_max_iterations}. "
                f"f(x) = {state.loss_val}, "
                f"|∇f(x)| = {state.gradient_norm} "
                f"|Δx| = {state.step_norm} "
                f"res. elps = {elapsed_residuals} "
                f"jac. elps = {elapsed_jacobian} "
                f"upd. elps = {elapsed_upd} "
            )
        if state.loss_val <= settings.loss_stop_threshold:
            state.stopping_reason = StoppingReason.ByLossValue
            break
        if state.gradient_norm <= settings.grad_norm_stop_threshold:
            state.stopping_reason = StoppingReason.ByGradNorm
            break
        if state.step_norm <= settings.step_norm_stop_threshold:
            state.stopping_reason = StoppingReason.ByStepNorm
            break
        state.variables_val += state.step_val

    # end of main loop
    if state.stopping_reason == StoppingReason.NotStopped:
        state.stopping_reason = StoppingReason.ByMaxIterations

    if settings.verbose:
        print(f"Optimization elapsed: {time.time() - start_time_optimization}")

    return state.variables_val, state
