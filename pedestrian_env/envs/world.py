import numpy as np

from pedestrian_env.envs.game_object import Car

class World:
    def __init__(self, width, height, pix_square_size, random):
        self.random = random

        self.agent_location = np.array([int(width / 2), height - 1], dtype=float)
        self.agent_target_location = self.agent_location
        self.initial_player_y = self.get_agent_cur_y()

        self.car_speeds_per_row = self._generate_rows(height)
        self.cars = self._generate_cars(width, height, pix_square_size)

    def update_positions(self, dt):
        agent_speed = 1.0
        tx, ty = self.agent_target_location
        cx, cy = self.agent_location

        dx = tx - cx
        dy = ty - cy
        if abs(dx) < 0.1:
            self.agent_location[0] = tx
            dx = 0
        if abs(dy) < 0.1:
            self.agent_location[1] = ty
            dy = 0
        if dx == 0 and dy == 0: return

        step_dist = agent_speed * (dt / 1000.0) # dt in ms
        dist = (dx ** 2 + dy ** 2) ** 0.5
        ratio = min(1.0, step_dist / dist)
        if dx != 0:
            self.agent_location[0] += dx * ratio
        if dy != 0:
            self.agent_location[1] += dy * ratio

    def get_agent_cur_y(self):
        return self.agent_location[1]

    def calculate_up_rewards(self):
        return self.initial_player_y - self.agent_location[1]

    def has_collided(self):
        for car in self.cars:
            if np.array_equal(self.agent_location, [car.x, car.y]):
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
