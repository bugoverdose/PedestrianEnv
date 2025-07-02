import math
from enum import Enum

class RowType(Enum):
    SAFE = 0
    CAR_GOING_RIGHT = 1
    CAR_GOING_LEFT = 2

class CrossWalk:
    def __init__(self, row1, row2, col):
        self.row1 = row1
        self.row2 = row2
        self.col = col
        self.is_active = False

# first-class object for simplicity
class CrossWalks:
    def __init__(self, elements):
        self.elements = elements

    @staticmethod
    def generate_crosswalks(agent, row_types, random):
        danger_zones = []
        danger_start_idx = None
        for row_idx in range(len(row_types)):
            row_type = row_types[row_idx]
            if danger_start_idx is None:
                if row_type != RowType.SAFE:
                    danger_start_idx = row_idx
                continue
            if row_type == RowType.SAFE:
                danger_zones.append([danger_start_idx, row_idx-1])
                danger_start_idx = None

        crosswalk_num = int(len(danger_zones) * 0.5)
        crosswalk_rows = random.choice(danger_zones, size=crosswalk_num, replace=False)
        crosswalks = []
        for i in range(crosswalk_num):
            [row1, row2] = crosswalk_rows[i]
            col = random.integers(agent.MIN_X, agent.MAX_X+1)
            crosswalks.append(CrossWalk(row1, row2, col))
        return CrossWalks(crosswalks)

    def update_activation(self, agent):
        agent_x, agent_y, agent_radius = agent.get_cur_pos()
        threshold_distance = agent_radius * 1.5
        for crosswalk in self.elements:
            start_y = crosswalk.row1 - 1
            end_y = crosswalk.row2 + 1
            if start_y <= agent_y <= end_y:
                distance = abs(agent_x - crosswalk.col) # check only row distance if in the same danger zone
            else:
                dy = min(abs(agent_y - start_y), abs(agent_y - end_y))
                dx = abs(agent_x - crosswalk.col)
                distance = math.hypot(dx, dy)
            crosswalk.is_active = distance <= threshold_distance
