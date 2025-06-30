import numpy as np

from pedestrian_env.envs.game_object import Agent, Car

class World:
    def __init__(self, grid_width, grid_height, pix_square_size, steps_per_second, random):
        self.random = random
        self.steps_per_second = steps_per_second
        self.agent = Agent(grid_width, grid_height, pix_square_size, steps_per_second)
        self.initial_player_y = self.agent.get_cur_y()

        self.car_speeds_per_row = self._generate_rows(grid_height)
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

    def _generate_rows(self, grid_height, max_safe_consecutive=3):
        rows = [0] # target area is safe
        safe_count = 1

        for _ in range(grid_height-2):
            choices = list([-1, 0, 1])
            if safe_count >= max_safe_consecutive:
                choices.remove(0)

            cur_row_type = self.random.choice(choices)
            rows.append(cur_row_type)

            if cur_row_type == 0:
                safe_count += 1
            else:
                safe_count = 0

        rows.append(0) # starting zone is safe
        return rows

    def _generate_cars(self, grid_width, grid_height, pix_square_size):
        cars = []
        for row_idx in range(grid_height):
            car_speed = self.car_speeds_per_row[row_idx]
            if car_speed == 0: continue
            car_type_seed = self.random.integers(0, 11, size=1, dtype=int)[0]
            if car_speed > 0:
                cars.append(Car(0, row_idx, 1, grid_width, pix_square_size, self.steps_per_second, car_type_seed=car_type_seed))
            elif car_speed < 0:
                cars.append(Car(grid_width - 1, row_idx, -1, grid_width, pix_square_size, self.steps_per_second, car_type_seed=car_type_seed))
        return cars
