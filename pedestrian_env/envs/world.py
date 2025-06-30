import math
from enum import Enum
import pygame

from pedestrian_env.envs.game_object import Agent, Car

class RowType(Enum):
    SAFE = 0
    DANGER = 1
    CAR_GOING_RIGHT = 2
    CAR_GOING_LEFT = 3

class World:
    def __init__(self, grid_width, grid_height, pix_square_size, steps_per_second, random, debug):
        self.random = random
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.pix_square_size = pix_square_size
        self.map_width = self.grid_width * self.pix_square_size
        self.map_height = self.grid_height * self.pix_square_size

        self.steps_per_second = steps_per_second
        self.agent = Agent(grid_width, grid_height, pix_square_size, steps_per_second, debug)
        self.initial_player_y = self.agent.get_cur_y()

        self.row_types, self.safe_row_idx_list, self.lane_boundary_idx_list = self._generate_rows(grid_height)
        self.cars = self._generate_cars(grid_width, grid_height, pix_square_size)

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
            pygame.draw.rect(canvas, (217, 217, 217), (0, safe_row_idx * self.pix_square_size - adjustment, self.map_width, self.pix_square_size))

        for boundary_idx in self.lane_boundary_idx_list:
            for x in range(0, self.grid_width, 3):
                pygame.draw.line(
                    canvas,
                    (250, 250, 250),
                    (x * self.pix_square_size, self.pix_square_size * boundary_idx + adjustment),
                    (x * self.pix_square_size + self.pix_square_size, self.pix_square_size * boundary_idx + adjustment),
                    width=3,
                )

        rendered_agent_position = self.agent.render(canvas)
        for car in self.cars:
            car.render(canvas)

        return canvas, rendered_agent_position

    def _generate_rows(self, grid_height, max_consecutive_danger_lanes=2):
        rows = []
        consecutive_danger_lanes = 0
        while len(rows) < grid_height - 2:
            available_rows = (grid_height - 2) - len(rows)
            if available_rows >= Car.HEIGHT and consecutive_danger_lanes < max_consecutive_danger_lanes:
                row_type = self.random.choice([RowType.SAFE, RowType.CAR_GOING_RIGHT, RowType.CAR_GOING_LEFT])
                if row_type != RowType.SAFE:
                    danger_zones = int((Car.HEIGHT - 1) / 2)
                    for _ in range(danger_zones):
                        rows.append(RowType.DANGER)
                    rows.append(row_type)
                    for _ in range(danger_zones):
                        rows.append(RowType.DANGER)
                    consecutive_danger_lanes += 1
                    continue
            rows.append(RowType.SAFE)
            consecutive_danger_lanes = 0

        total_rows = [RowType.SAFE] + rows + [RowType.SAFE] # target area + lanes + starting area

        safe_row_idx_list = []
        for idx in range(len(total_rows)):
            if total_rows[idx] == RowType.SAFE:
                safe_row_idx_list.append(idx)

        lane_boundary_idx_list = []
        for idx in range(1, len(total_rows)):
            if total_rows[idx] != RowType.SAFE:
                lane_boundary_idx_list.append(idx)
            elif total_rows[idx] == RowType.SAFE and total_rows[idx-1] != RowType.SAFE:
                lane_boundary_idx_list.pop()

        return total_rows, safe_row_idx_list, lane_boundary_idx_list

    def _generate_cars(self, grid_width, grid_height, pix_square_size):
        cars = []
        for row_idx in range(grid_height):
            initial_x = self.random.integers(0, grid_width)
            if self.row_types[row_idx] == RowType.CAR_GOING_RIGHT:
                cars.append(Car(initial_x, row_idx, 1, grid_width, pix_square_size, self.steps_per_second))
            elif self.row_types[row_idx] == RowType.CAR_GOING_LEFT:
                cars.append(Car(initial_x, row_idx, -1, grid_width, pix_square_size, self.steps_per_second))
        return cars

    def _check_collision(self, car):
        cx, cy, radius = self.agent.get_cur_pos() # circle
        left_x, right_x, top_y, bottom_y = car.get_cur_pos() # rectangle
        closest_x = max(left_x, min(cx, right_x))
        closest_y = max(top_y, min(cy, bottom_y))
        distance = math.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)
        return distance < radius
