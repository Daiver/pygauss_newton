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
        settings: Settings
) -> StoppingReason:
    pass
