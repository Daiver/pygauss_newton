import time
from typing import Callable, Union
import numpy as np

from .settings import Settings
from .utils import time_fn
from .stopping_reason import StoppingReason, stopping_condition_by_state_values


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

        self.mu_multiplier = None
        self.mu_value = None

        self.stopping_reason = StoppingReason.NotStopped


def compute_predicted_gain(gradient_val: np.ndarray, step_val: np.ndarray, current_mu: float) -> float:
    tmp = current_mu * step_val - gradient_val
    return 0.5 * tmp.dot(step_val)


def levenberg_marquardt(
        residuals_func: Callable,
        jacobian_func: Callable,
        x0: Union[np.ndarray],
        settings: Settings = None,
        update_functor: Callable = None
) -> (np.ndarray, OptimizationState):
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

    epsilon_for_division = 1e-9
    state.mu_multiplier = 2.0

    # TODO: move it to settings
    initial_mu_scale = 1e-6
    n_maximum_search_iterations = 10

    state.residuals_val, elapsed_residuals = time_fn(residuals_func, state.variables_val)
    state.jacobian_val, elapsed_jacobian = time_fn(jacobian_func, state.variables_val)

    state.gradient_val = state.jacobian_val.T @ state.residuals_val
    state.gradient_norm = np.linalg.norm(state.gradient_val)
    state.loss_val = 0.5 * state.residuals_val.T @ state.residuals_val

    state.hessian_val = state.jacobian_val.T @ state.jacobian_val

    state.mu_value = initial_mu_scale * np.max(np.abs(np.diag(state.hessian_val)))

    for iter_ind in range(settings.n_max_iterations):
        state.iter_ind = iter_ind

        if settings.verbose:
            print(
                f"{iter_ind + 1}/{settings.n_max_iterations}. "
                f"f(x) = {state.loss_val}, "
                f"|∇f(x)| = {state.gradient_norm} "
                f"|Δx| = {state.step_norm} "
                f"μ = {state.mu_value}"
            )

        for search_iter in range(n_maximum_search_iterations):
            dumped_hessian = state.hessian_val + state.mu_value * eye
            state.step_val = -np.linalg.solve(dumped_hessian, state.gradient_val)
            state.step_norm = np.linalg.norm(state.step_val)

            new_variables_val = state.variables_val + state.step_val
            new_residuals_val, elapsed_residuals = time_fn(residuals_func, new_variables_val)
            new_loss_val = 0.5 * new_residuals_val.T @ new_residuals_val

            state.stopping_reason = stopping_condition_by_state_values(
                settings=settings, loss_val=new_loss_val, gradient_norm=state.gradient_norm, step_norm=state.step_norm
            )
            if state.stopping_reason != StoppingReason.NotStopped:
                state.variables_val = new_variables_val
                state.residuals_val = new_residuals_val
                state.loss_val = new_loss_val
                break

            real_gain = state.loss_val - new_loss_val
            predicted_gain = compute_predicted_gain(state.gradient_val, state.step_val, state.mu_value)
            gain_ratio = real_gain / (predicted_gain + epsilon_for_division)
            # print(
            #     f"new_loss {new_loss_val} "
            #     f"predicted_gain {predicted_gain}, "
            #     f"real_gain {real_gain}, "
            #     f"gain_ratio {gain_ratio}"
            # )
            if gain_ratio > 0.0:
                state.variables_val = new_variables_val
                state.residuals_val = new_residuals_val
                state.loss_val = new_loss_val

                state.mu_value = state.mu_value * max(1.0 / 3.0, 1.0 - (2 * gain_ratio) ** 3)
                state.mu_multiplier = 2.0

                state.jacobian_val, elapsed_jacobian = time_fn(jacobian_func, state.variables_val)

                state.gradient_val = state.jacobian_val.T @ state.residuals_val
                state.gradient_norm = np.linalg.norm(state.gradient_val)
                state.loss_val = 0.5 * state.residuals_val.T @ state.residuals_val
                state.hessian_val = state.jacobian_val.T @ state.jacobian_val

                elapsed_upd = 0
                if update_functor is not None:
                    functor_result, elapsed_upd = time_fn(update_functor, state.variables_val, state)
                    if functor_result is False:
                        state.stopping_reason = StoppingReason.ByCallback
                        break

                break
            state.mu_value = state.mu_value * state.mu_multiplier
            state.mu_multiplier = 2 * state.mu_multiplier

        if state.stopping_reason != StoppingReason.NotStopped:
            break

    if settings.verbose:
        print(f"Optimization elapsed: {time.time() - start_time_optimization}")

    return state.variables_val, state
