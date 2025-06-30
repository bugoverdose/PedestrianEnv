from enum import Enum
import numpy as np

class Action(Enum):
    nothing = 0
    up = 1
    down = 2
    right = 3
    left = 4

ACTION_TO_DELTA = {
    Action.nothing: np.array([0, 0]),
    Action.up: np.array([0, -1]),
    Action.down: np.array([0, 1]),
    Action.right: np.array([1, 0]),
    Action.left: np.array([-1, 0]),
}
