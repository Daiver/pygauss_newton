from enum import Enum


class StoppingReason(Enum):
    NotStopped = -1
    ByCallback = 0
    ByLossValue = 1
    ByGradNorm = 2
    ByStepNorm = 3
