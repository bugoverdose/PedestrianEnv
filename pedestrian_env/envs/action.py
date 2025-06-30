from enum import Enum
import numpy as np

class Action(Enum):
    NOTHING = 0
    UP = 1
    DOWN = 2
    RIGHT = 3
    LEFT = 4

ACTION_TO_DELTA = {
    Action.NOTHING: np.array([0, 0]),
    Action.UP: np.array([0, -1]),
    Action.DOWN: np.array([0, 1]),
    Action.RIGHT: np.array([1, 0]),
    Action.LEFT: np.array([-1, 0]),
}
