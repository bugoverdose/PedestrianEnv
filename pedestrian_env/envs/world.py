import math
from enum import Enum
import pygame

from pedestrian_env.envs.game_object import Agent, Car

class RowType(Enum):
    SAFE = 0
    CAR_GOING_RIGHT = 1
    CAR_GOING_LEFT = 2

class World:
    def __init__(self, map_grid_width, map_grid_height, pix_square_size, steps_per_second, random, debug):
        self.random = random
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.pix_square_size = pix_square_size
        self.map_width = self.map_grid_width * self.pix_square_size
        self.map_height = self.map_grid_height * self.pix_square_size

        self.steps_per_second = steps_per_second
        self.agent = Agent(map_grid_width, map_grid_height, pix_square_size, steps_per_second, debug)
        self.initial_player_y = self.agent.get_cur_y()

        self._generate_rows(map_grid_height)
        self.cars = self._generate_cars(map_grid_width, map_grid_height, pix_square_size)

    def target_lane_reached(self):
        cur_y = self.agent.get_cur_y()
        return cur_y == Agent.TARGET_LANE

    def update_positions(self, dt):
        self.agent.update_position(dt)
        for car in self.cars:
            car.update_position(dt)

    def calculate_up_rewards(self):
        return int(self.steps_per_second * (self.initial_player_y - self.agent.get_target_y()))

    def has_collided(self):
        for car in self.cars:
            if self._check_collision(car):
                return True
        return False

    def render(self):
        canvas = pygame.Surface((self.map_width, self.map_height))
        canvas.fill((89, 89, 89))

        adjustment = 0.5 * self.pix_square_size
        for safe_row_idx in self.safe_row_idx_list:
            pygame.draw.rect(canvas, (217, 217, 217), (0, safe_row_idx * self.pix_square_size - adjustment, self.map_width, self.pix_square_size + 1))

        for boundary_idx in self.lane_boundary_idx_list:
            for x in range(0, self.map_grid_width, 5):
                start_x = x * self.pix_square_size
                pygame.draw.line(
                    canvas,
                    (250, 250, 250),
                    (start_x, self.pix_square_size * boundary_idx + adjustment),
                    (start_x + 2 * self.pix_square_size, self.pix_square_size * boundary_idx + adjustment),
                    width=3,
                )

        rendered_agent_position = self.agent.render(canvas)
        for car in self.cars:
            car.render(canvas)

        return canvas, rendered_agent_position

    def _generate_rows(self, map_grid_height, max_consecutive_danger_lanes=4):
        rows = []
        consecutive_danger_lanes = 0
        while len(rows) < map_grid_height - 2:
            available_rows = (map_grid_height - 2) - len(rows)
            if available_rows >= Car.MAX_HEIGHT and consecutive_danger_lanes < max_consecutive_danger_lanes:
                row_type = self.random.choice([RowType.SAFE, 1, 2])
                if row_type == 1:
                    rows.append(RowType.CAR_GOING_RIGHT)
                    rows.append(RowType.CAR_GOING_RIGHT)
                    consecutive_danger_lanes += 2
                    continue
                elif row_type == 2:
                    rows.append(RowType.CAR_GOING_LEFT)
                    rows.append(RowType.CAR_GOING_LEFT)
                    consecutive_danger_lanes += 2
                    continue
            rows.append(RowType.SAFE)
            consecutive_danger_lanes = 0

        total_rows = [RowType.SAFE] + rows + [RowType.SAFE] # target area + lanes + starting area

        self.safe_row_idx_list = []
        for idx in range(len(total_rows)):
            if total_rows[idx] == RowType.SAFE:
                self.safe_row_idx_list.append(idx)

        self.lane_boundary_idx_list = []
        for idx in range(len(total_rows)-1):
            if total_rows[idx] != RowType.SAFE and total_rows[idx+1] != RowType.SAFE:
                self.lane_boundary_idx_list.append(idx)

        self.max_height_dic = {}
        for row_idx in range(len(total_rows)):
            if total_rows[row_idx] == RowType.SAFE: continue
            self.max_height_dic[row_idx] = 1
            prev_row_idx = row_idx - 1
            while prev_row_idx >= 0:
                if total_rows[prev_row_idx] != total_rows[row_idx]: break
                self.max_height_dic[prev_row_idx] = min(self.max_height_dic[prev_row_idx] + 1, Car.MAX_HEIGHT)
                prev_row_idx -= 1
        self.row_types = total_rows

    def _generate_cars(self, map_grid_width, map_grid_height, pix_square_size):
        cars = []
        for row_idx in range(map_grid_height):
            if self.row_types[row_idx] == RowType.SAFE: continue
            initial_x = self.random.integers(0, map_grid_width)
            height = self.random.choice(list(range(1, self.max_height_dic[row_idx] + 1)))
            width = self.random.choice(Car.CAR_SIZES[height])[1]
            speed = 1 if self.row_types[row_idx] == RowType.CAR_GOING_RIGHT else -1
            row_idx += (height - 1) * 0.5
            cars.append(Car(initial_x, row_idx, width, height, speed, map_grid_width, pix_square_size, self.steps_per_second))

        self.random.shuffle(cars)
        non_overlapping_cars = []
        for car in cars:
            if all(not self._check_overlapping(car, other) for other in non_overlapping_cars):
                non_overlapping_cars.append(car)
        return non_overlapping_cars

    def _check_overlapping(self, car1, car2):
        l1, r1, t1, b1 = car1.get_cur_pos()
        l2, r2, t2, b2 = car2.get_cur_pos()
        if r1 <= l2 or r2 <= l1:
            return False
        if b1 <= t2 or b2 <= t1:
            return False
        return True

    def _check_collision(self, car):
        cx, cy, radius = self.agent.get_cur_pos() # circle
        left_x, right_x, top_y, bottom_y = car.get_cur_pos() # rectangle
        closest_x = max(left_x, min(cx, right_x))
        closest_y = max(top_y, min(cy, bottom_y))
        distance = math.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)
        return distance < radius
