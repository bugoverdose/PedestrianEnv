import math
from enum import Enum

class RowType(Enum):
    SAFE = 0
    CAR_GOING_RIGHT = 1
    CAR_GOING_LEFT = 2

class CrossWalk:
    RATIO = 0.6

    def __init__(self, col, threshold_distance):
        self.col = col
        self.threshold_distance = threshold_distance
        self.is_active = False

    def get_left_right(self):
        left = self.col - self.threshold_distance
        right = self.col + self.threshold_distance
        return left, right

    def __str__(self):
        return f"CrossWalk(active={self.is_active}, col={self.col})"
