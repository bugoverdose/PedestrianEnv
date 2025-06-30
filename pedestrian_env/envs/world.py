import numpy as np

from pedestrian_env.envs.game_object import Agent, Car

class World:
    def __init__(self, width, height, pix_square_size, random):
        self.random = random
        self.agent = Agent(width, height, pix_square_size)
        self.initial_player_y = self.agent.get_cur_y()

        self.car_speeds_per_row = self._generate_rows(height)
        self.cars = self._generate_cars(width, height, pix_square_size)

    def target_lane_reached(self):
        cur_y = self.agent.get_cur_y()
        return cur_y == 0

    def update_positions(self, dt):
        self.agent.update_position(dt)

    def calculate_up_rewards(self):
        return self.initial_player_y - self.agent.get_cur_y()

    # TODO: fix bug after vehicle re-implementation
    def has_collided(self):
        for car in self.cars:
            if np.array_equal(self.agent.cur_location, [car.x, car.y]):
                return True
        return False

    def _generate_rows(self, height, max_safe_consecutive=3):
        rows = [0] # target area is safe
        safe_count = 1

        for _ in range(height-2):
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

    def _generate_cars(self, width, height, pix_square_size):
        cars = []
        for row_idx in range(height):
            car_speed = self.car_speeds_per_row[row_idx]
            if car_speed == 0: continue
            car_type_seed = self.random.integers(0, 11, size=1, dtype=int)[0]
            if car_speed > 0:
                cars.append(Car(0, row_idx, 1, pix_square_size, car_type_seed=car_type_seed))
            elif car_speed < 0:
                cars.append(Car(width - 1, row_idx, -1, pix_square_size, car_type_seed=car_type_seed))
        return cars
