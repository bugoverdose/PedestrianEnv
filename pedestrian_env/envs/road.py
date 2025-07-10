import math
from enum import Enum

from pedestrian_env.envs.crosswalk import CrossWalk

class RowType(Enum):
    SAFE = 0
    CAR_GOING_RIGHT = 1
    CAR_GOING_LEFT = 2

class Roads:
    def __init__(self, agent, camera_size, row_types, random):
        self.agent = agent
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

        roads = []
        agent_initial_col, _, agent_radius = agent.get_cur_pos()
        out_of_sight_buffer = camera_size/2 + 1
        left = random.random() >= 0.5
        for candidate in candidates:
            [row1, row2] = candidate
            if random.random() >= CrossWalk.RATIO: 
                roads.append(Road(row1, row2, None, random))
                continue
            crosswalk_range = (agent.MIN_X, agent_initial_col - out_of_sight_buffer) if left else (agent_initial_col + out_of_sight_buffer + 1, agent.MAX_X+1)
            left = not left
            col = random.integers(crosswalk_range[0], crosswalk_range[1])
            roads.append(Road(row1, row2, CrossWalk(col, agent_radius * 1.5), random))

        self.elements = roads

    def update_crosswalk_activation(self):
        agent_x, agent_y, _ = self.agent.get_cur_pos()
        for road in self.elements:
            crosswalk = road.crosswalk
            if crosswalk is None: continue
            start_y = road.row1 - 1
            end_y = road.row2 + 1
            if start_y <= agent_y <= end_y:
                distance = abs(agent_x - crosswalk.col) # check only row distance if in the same danger zone
            else:
                dy = min(abs(agent_y - start_y), abs(agent_y - end_y))
                dx = abs(agent_x - crosswalk.col)
                distance = math.hypot(dx, dy)
            crosswalk.is_active = distance <= crosswalk.threshold_distance

class Road:
    PENALTIES = [100, 500, 1000]

    def __init__(self, row1, row2, crosswalk, random):
        self.row1 = row1
        self.row2 = row2
        self.penalty = random.choice(self.PENALTIES)
        self.crosswalk = crosswalk
