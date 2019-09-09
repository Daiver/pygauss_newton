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
        self.stopping_reason = StoppingReason.NotStopped


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

    for iter_ind in range(settings.n_max_iterations):
        pass

    raise NotImplementedError
