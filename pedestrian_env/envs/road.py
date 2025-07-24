import math
from enum import Enum

import pygame

from pedestrian_env.envs.car_details import CarColorType, get_max_car_grid_height
from pedestrian_env.envs.utils import is_overlapping
from pedestrian_env.envs.car_details import _RiskDetails

class RowType(Enum):
    SAFE = 0
    CAR_GOING_RIGHT = 1
    CAR_GOING_LEFT = 2

class Roads:
    SAFE_END_LANE_COUNT = 1
    SAFE_START_LANE_COUNT = 1
    MIN_ROAD_SIZE = 2
    MAX_ROAD_SIZE = 4
    ROAD_GRAY_COLOR = (89, 89, 89)
    ROAD_WHITE_COLOR = (250, 250, 250)
    SAFE_WHITE_COLOR = (217, 217, 217)

    def __init__(self, agent, camera_size, map_grid_height, random):
        max_car_grid_height = get_max_car_grid_height()
        if max_car_grid_height > self.MIN_ROAD_SIZE:
            raise Exception(f"maximum car height can not be bigger than minimum road size {max_car_grid_height} > {self.MIN_ROAD_SIZE}")

        row_types = [RowType.SAFE] * map_grid_height
        consecutive_danger_lanes = 0
        consecutive_safe_lanes = 1
        cur_row_idx = self.SAFE_END_LANE_COUNT
        while cur_row_idx < map_grid_height:
            # go back to add roads on too much consecutive safe zones
            if consecutive_safe_lanes >= self.MIN_ROAD_SIZE + 1:
                go_back = self.MIN_ROAD_SIZE
                cur_row_idx -= go_back
                consecutive_safe_lanes -= go_back

            available_rows = map_grid_height - self.SAFE_START_LANE_COUNT - cur_row_idx # save safe zones near start lane
            if available_rows <= 0: break 
            if available_rows >= self.MIN_ROAD_SIZE and consecutive_danger_lanes < self.MAX_ROAD_SIZE:
                if cur_row_idx == 1 or available_rows == self.MIN_ROAD_SIZE:
                    # reduce consecutive safe zones near the start and end
                    row_type = random.choice([RowType.CAR_GOING_RIGHT, RowType.CAR_GOING_LEFT])
                else:
                    row_type = random.choice([RowType.SAFE, RowType.CAR_GOING_RIGHT, RowType.CAR_GOING_LEFT])
                if row_type != RowType.SAFE:
                    for _ in range(self.MIN_ROAD_SIZE):
                        row_types[cur_row_idx] = row_type
                        consecutive_danger_lanes += 1
                        consecutive_safe_lanes = 0
                        cur_row_idx += 1
                    continue
            consecutive_danger_lanes = 0
            consecutive_safe_lanes += 1
            cur_row_idx += 1

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
                self.max_height_dic[prev_row_idx] = min(self.max_height_dic[prev_row_idx] + 1, max_car_grid_height)
                prev_row_idx -= 1
        
        uid = 0
        roads = []
        danger_start_idx = None
        for row_idx in range(len(row_types)):
            row_type = row_types[row_idx]
            if danger_start_idx is None:
                if row_type != RowType.SAFE:
                    danger_start_idx = row_idx
                continue
            if row_type == RowType.SAFE:
                uid += 1
                row1 = danger_start_idx
                row2 = row_idx-1
                roads.append(Road(uid, row1, row2, row_types[row1:row2+1], random))
                danger_start_idx = None

        # add crosswalks on road
        [agent_initial_col, _] = agent.cur_location
        buffer = int(camera_size * 0.4) # some are visible from the starting point
        left = random.random() >= 0.5
        for road in roads:
            if random.random() >= CrossWalk.RATIO: continue
            crosswalk_range = (agent.MIN_X, max(agent.MIN_X+1, agent_initial_col - buffer)) if left else (min(agent.MAX_X, agent_initial_col + buffer), agent.MAX_X+1)
            left = not left
            col = random.integers(crosswalk_range[0], crosswalk_range[1])
            road.crosswalk = CrossWalk(col, road.row1, road.row2)

        self.agent = agent
        self.elements = roads

    def update_crosswalk_activation(self):
        [agent_x, agent_y] = self.agent.cur_location
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
            crosswalk.is_active = distance <= CrossWalk.THRESHOLD_DISTANCE

    def render(self, background, map_grid_width, pix_square_size):
        map_width = map_grid_width * pix_square_size
        adjustment = 0.5 * pix_square_size

        background.fill(self.ROAD_GRAY_COLOR)
        for safe_row_idx in self.safe_row_idx_list:
            pygame.draw.rect(background, self.SAFE_WHITE_COLOR, (0, safe_row_idx * pix_square_size - adjustment, map_width, pix_square_size + 1))

        for other_direction_idx in self.other_direction_boundary_idx_list:
            start_x = 0
            pygame.draw.line(
                background,
                self.ROAD_WHITE_COLOR,
                (start_x, pix_square_size * other_direction_idx + adjustment),
                (map_width, pix_square_size * other_direction_idx + adjustment),
                width=3,
            )

        for boundary_idx in self.lane_boundary_idx_list:
            for x in range(0, map_grid_width, 5):
                start_x = x * pix_square_size
                pygame.draw.line(
                    background,
                    self.ROAD_WHITE_COLOR,
                    (start_x, pix_square_size * boundary_idx + adjustment),
                    (start_x + 2 * pix_square_size, pix_square_size * boundary_idx + adjustment),
                    width=6,
                )

        # add crosswalks
        for road in self.elements:
            crosswalk = road.crosswalk
            if crosswalk is None: continue
            start_x, end_x = crosswalk.get_visible_left_right()
            cw_width = (end_x - start_x) * pix_square_size
            start_x = start_x * pix_square_size
            start_y = road.row1 * pix_square_size - adjustment
            end_y = road.row2 * pix_square_size + adjustment
            cw_height = end_y - start_y
            # NOTE: cover up background (-1 pixel at top and bottom, +pix_square_size at left and right)
            pygame.draw.rect(background, self.ROAD_GRAY_COLOR, (start_x - pix_square_size, start_y + 1, cw_width + pix_square_size * 2, cw_height - 1))

            stripe_count = 3 * (road.row2 - road.row1 + 1)
            stripe_thickness = pix_square_size / 3
            for i in range(stripe_count):
                for j in range(2):
                    if i % 2 == j: continue
                    stripe_rect = pygame.Rect(start_x + (cw_width/2 * j), start_y + (i * stripe_thickness), cw_width/2, stripe_thickness)
                    pygame.draw.rect(background, self.ROAD_WHITE_COLOR, stripe_rect)

class Road:
    def __init__(self, uid, row1, row2, row_types, random):
        self.uid = uid
        self.row1 = row1
        self.row2 = row2
        self.going_right = [row_type == RowType.CAR_GOING_RIGHT for row_type in row_types]
        self.car_color_type = random.choice([CarColorType.RED, CarColorType.YELLOW, CarColorType.GREEN])
        self.crosswalk = None

    def observe(self, agent, cars, crosswalk_visible_range, game_screen_visible_range):
        [agent_x, agent_y] = agent.cur_location

        agent_head_y, agent_tail_y = agent_y - agent.RADIUS, agent_y + agent.RADIUS
        cur_road_up_y = self.row1 - 0.5
        cur_road_end_dist = agent_tail_y - cur_road_up_y

        cur_crosswalk_discovered = 0 # not found
        cur_crosswalk_x_diff = 0
        if self.crosswalk is not None:
            x_dist = self.crosswalk.col - agent_x
            if abs(x_dist) < crosswalk_visible_range:
                cur_crosswalk_discovered = 1 # found
                cur_crosswalk_x_diff = x_dist

        agent_left_x, agent_right_x = agent_x - agent.RADIUS, agent_x + agent.RADIUS

        nearby_car_infos = {}
        for i in range(self.row2 - self.row1 + 1):
            row_idx = self.row1 + i
            if self.going_right[i]:
                nearby_car_infos[row_idx] = [-game_screen_visible_range, 0] # stopped at left end, heading right
            else:
                nearby_car_infos[row_idx] = [game_screen_visible_range, 0] # stopped at right end, heading left
        for car in cars.elements:
            if car.road.uid != self.uid: continue
            car_left_x, car_right_x = car.get_cur_x_pos()
            if abs(agent_x - car_left_x) >= game_screen_visible_range and abs(agent_x - car_right_x) >= game_screen_visible_range:
                continue # out of sight
            agent.road_penalty_dict[self.uid] = car.car_detail.penalty

            if is_overlapping(agent_left_x, agent_right_x, car_left_x, car_right_x):
                x_dist = 0
            elif car_right_x < agent_left_x:
                x_dist = car_right_x - agent_left_x
            elif agent_right_x < car_left_x:
                x_dist = car_left_x - agent_right_x
            else:
                raise Exception("invalid implementation")

            for row_idx in car.row_indices:
                closest_car_x_dist = nearby_car_infos[row_idx][0]
                if closest_car_x_dist == 0: continue # another blocked
                if car.cur_speed > 0 and x_dist > 0: continue # moving away
                if car.cur_speed < 0 and x_dist < 0: continue # moving away
                if abs(closest_car_x_dist) < abs(x_dist): continue # farther away
                # save closest and most dangerous car position
                actual_car_speed = car.cur_speed if car.is_moving else 0
                nearby_car_infos[row_idx] = [x_dist, actual_car_speed]

        # closest lane to farthest + flatten out list
        nearby_cars = []
        for row_idx in range(self.row2, self.row1 - 1, -1):
            nearby_cars.append(nearby_car_infos[row_idx][0])
            nearby_cars.append(nearby_car_infos[row_idx][1])

        return cur_road_end_dist, cur_crosswalk_x_diff, cur_crosswalk_discovered, nearby_cars

class CrossWalk:
    RATIO = 0.6
    THRESHOLD_DISTANCE = 1.2
    VISIBLE_WIDTH = 1

    def __init__(self, col, start_row, end_row):
        self.col = col
        self.left_end = self.col - self.VISIBLE_WIDTH
        self.right_end = self.col + self.VISIBLE_WIDTH
        self.start_row = start_row
        self.end_row = end_row
        self.is_active = False

    def get_activation_left_right(self):
        left = self.col - self.THRESHOLD_DISTANCE
        right = self.col + self.THRESHOLD_DISTANCE
        return left, right

    def get_visible_left_right(self):
        return self.left_end, self.right_end

    def __str__(self):
        return f"CrossWalk(active={self.is_active}, col={self.col})"
