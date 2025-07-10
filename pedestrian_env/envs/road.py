import math
from enum import Enum

from pedestrian_env.envs.game_object import Car

class RowType(Enum):
    SAFE = 0
    CAR_GOING_RIGHT = 1
    CAR_GOING_LEFT = 2

class Roads:
    MAX_ROAD_SIZE = 4
    def __init__(self, agent, camera_size, map_grid_height, random):
        rows = []
        consecutive_danger_lanes = 0
        while len(rows) < map_grid_height - 2:
            available_rows = (map_grid_height - 2) - len(rows)
            if available_rows >= Car.MAX_HEIGHT and consecutive_danger_lanes < self.MAX_ROAD_SIZE:
                row_type = random.choice([RowType.SAFE, RowType.CAR_GOING_RIGHT, RowType.CAR_GOING_LEFT])
                if row_type != RowType.SAFE:
                    rows.append(row_type)
                    rows.append(row_type)
                    consecutive_danger_lanes += 2
                    continue
            rows.append(RowType.SAFE)
            consecutive_danger_lanes = 0
        row_types = [RowType.SAFE] + rows + [RowType.SAFE] # target area + lanes + starting area

        self.safe_row_idx_list = []
        for idx in range(len(row_types)):
            if row_types[idx] == RowType.SAFE:
                self.safe_row_idx_list.append(idx)

        self.lane_boundary_idx_list = []
        for idx in range(len(row_types)-1):
            if row_types[idx] != RowType.SAFE and row_types[idx+1] != RowType.SAFE:
                self.lane_boundary_idx_list.append(idx)

        self.other_direction_boundary_idx_list = []
        for idx in range(len(row_types)-1):
            if row_types[idx] == RowType.CAR_GOING_RIGHT and row_types[idx+1] == RowType.CAR_GOING_LEFT:
                self.other_direction_boundary_idx_list.append(idx)
                self.lane_boundary_idx_list.remove(idx)
            if row_types[idx] == RowType.CAR_GOING_LEFT and row_types[idx+1] == RowType.CAR_GOING_RIGHT:
                self.other_direction_boundary_idx_list.append(idx)
                self.lane_boundary_idx_list.remove(idx)

        self.max_height_dic = {}
        for row_idx in range(len(row_types)):
            if row_types[row_idx] == RowType.SAFE: continue
            self.max_height_dic[row_idx] = 1
            prev_row_idx = row_idx - 1
            while prev_row_idx >= 0:
                if row_types[prev_row_idx] != row_types[row_idx]: break
                self.max_height_dic[prev_row_idx] = min(self.max_height_dic[prev_row_idx] + 1, Car.MAX_HEIGHT)
                prev_row_idx -= 1
        
        roads = []
        danger_start_idx = None
        for row_idx in range(len(row_types)):
            row_type = row_types[row_idx]
            if danger_start_idx is None:
                if row_type != RowType.SAFE:
                    danger_start_idx = row_idx
                continue
            if row_type == RowType.SAFE:
                row1 = danger_start_idx
                row2 = row_idx-1
                roads.append(Road(row1, row2, row_types[row1:row2+1], random))
                danger_start_idx = None

        # add crosswalk on road
        agent_initial_col, _, agent_radius = agent.get_cur_pos()
        out_of_sight_buffer = camera_size/2 + 1
        left = random.random() >= 0.5
        for road in roads:
            if random.random() >= CrossWalk.RATIO: continue
            crosswalk_range = (agent.MIN_X, agent_initial_col - out_of_sight_buffer) if left else (agent_initial_col + out_of_sight_buffer + 1, agent.MAX_X+1)
            left = not left
            col = random.integers(crosswalk_range[0], crosswalk_range[1])
            road.crosswalk = CrossWalk(col, agent_radius * 1.5)

        self.agent = agent
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

    def __init__(self, row1, row2, row_types, random):
        self.row1 = row1
        self.row2 = row2
        self.going_right = [row_type == RowType.CAR_GOING_RIGHT for row_type in row_types]
        self.penalty = random.choice(self.PENALTIES)
        self.crosswalk = None

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
