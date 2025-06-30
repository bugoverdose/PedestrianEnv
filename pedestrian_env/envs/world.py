from enum import Enum
import numpy as np

from pedestrian_env.envs.game_object import Agent, Car

class RowType(Enum):
    SAFE = 0
    DANGER = 1
    CAR_GOING_RIGHT = 2
    CAR_GOING_LEFT = 3

class World:
    def __init__(self, grid_width, grid_height, pix_square_size, steps_per_second, random):
        self.random = random
        self.steps_per_second = steps_per_second
        self.agent = Agent(grid_width, grid_height, pix_square_size, steps_per_second)
        self.initial_player_y = self.agent.get_cur_y()

        self.row_types = self._generate_rows(grid_height)
        self.cars = self._generate_cars(grid_width, grid_height, pix_square_size)

    def target_lane_reached(self):
        cur_y = self.agent.get_cur_y()
        return cur_y == 0

    def update_positions(self, dt):
        self.agent.update_position(dt)
        for car in self.cars:
            car.update_position(dt)

    def calculate_up_rewards(self):
        return int(self.steps_per_second * (self.initial_player_y - self.agent.get_target_y()))

    # TODO: fix bug after vehicle re-implementation
    def has_collided(self):
        for car in self.cars:
            if np.array_equal(self.agent.cur_location, [car.cur_location[0], car.cur_location[1]]):
                return True
        return False

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
        return [RowType.SAFE] + rows + [RowType.SAFE] # target area + lanes + starting area

    def _generate_cars(self, grid_width, grid_height, pix_square_size):
        cars = []
        for row_idx in range(grid_height):
            initial_x = self.random.integers(0, grid_width)
            if self.row_types[row_idx] == RowType.CAR_GOING_RIGHT:
                cars.append(Car(initial_x, row_idx, 1, grid_width, pix_square_size, self.steps_per_second))
            elif self.row_types[row_idx] == RowType.CAR_GOING_LEFT:
                cars.append(Car(initial_x, row_idx, -1, grid_width, pix_square_size, self.steps_per_second))
        return cars
