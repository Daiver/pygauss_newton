from enum import Enum

from .settings import Settings


class StoppingReason(Enum):
    NotStopped = -1
    ByCallback = 0
    ByLossValue = 1
    ByGradNorm = 2
    ByStepNorm = 3
    ByMaxIterations = 4


def stopping_condition_by_state_values(
        settings: Settings,
        loss_val: float,
        gradient_norm: float,
        step_norm: float
) -> StoppingReason:
    if loss_val <= settings.loss_stop_threshold:
        return StoppingReason.ByLossValue
    if gradient_norm <= settings.grad_norm_stop_threshold:
        return StoppingReason.ByGradNorm
    if step_norm <= settings.step_norm_stop_threshold:
        return StoppingReason.ByStepNorm

    return StoppingReason.NotStopped
