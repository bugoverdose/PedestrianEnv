import math
from enum import Enum

class RowType(Enum):
    SAFE = 0
    CAR_GOING_RIGHT = 1
    CAR_GOING_LEFT = 2

class CrossWalk:
    RATIO = 0.6

    def __init__(self, row1, row2, col, threshold_distance):
        self.row1 = row1
        self.row2 = row2
        self.col = col
        self.threshold_distance = threshold_distance
        self.is_active = False

    def get_left_right(self):
        left = self.col - self.threshold_distance
        right = self.col + self.threshold_distance
        return left, right

    def __str__(self):
        return f"CrossWalk(active={self.is_active}, row=({self.row1}~{self.row2}), col={self.col})"

# first-class object for simplicity
class CrossWalks:

    def __init__(self, agent, elements):
        self.agent = agent
        self.elements = elements

    @staticmethod
    def generate_crosswalks(agent, camera_size, row_types, random):
        candidates = []
        danger_start_idx = None
        for row_idx in range(len(row_types)):
            row_type = row_types[row_idx]
            if danger_start_idx is None:
                if row_type != RowType.SAFE:
                    danger_start_idx = row_idx
                continue
            if row_type == RowType.SAFE:
                candidates.append([danger_start_idx, row_idx-1])
                danger_start_idx = None

        crosswalks = []
        agent_initial_col, _, agent_radius = agent.get_cur_pos()
        out_of_sight_buffer = camera_size/2 + 1
        left = random.random() >= 0.5
        for candidate in candidates:
            if random.random() >= CrossWalk.RATIO: continue
            [row1, row2] = candidate
            if left:
                col = random.integers(agent.MIN_X, agent_initial_col - out_of_sight_buffer)
            else:
                col = random.integers(agent_initial_col + out_of_sight_buffer + 1, agent.MAX_X+1)
            crosswalks.append(CrossWalk(row1, row2, col, agent_radius * 1.5))
            left = not left
        return CrossWalks(agent, crosswalks)

    def update_activation(self):
        agent_x, agent_y, _ = self.agent.get_cur_pos()
        for crosswalk in self.elements:
            start_y = crosswalk.row1 - 1
            end_y = crosswalk.row2 + 1
            if start_y <= agent_y <= end_y:
                distance = abs(agent_x - crosswalk.col) # check only row distance if in the same danger zone
            else:
                dy = min(abs(agent_y - start_y), abs(agent_y - end_y))
                dx = abs(agent_x - crosswalk.col)
                distance = math.hypot(dx, dy)
            crosswalk.is_active = distance <= crosswalk.threshold_distance
