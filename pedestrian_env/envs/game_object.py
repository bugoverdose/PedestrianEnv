import pygame
import numpy as np

class GameObject:

    def __init__(self, speed, cur_location, pix_square_size, steps_per_second):
        self.speed = speed
        self.cur_location = cur_location
        self.target_location = self.cur_location
        self.pix_square_size = pix_square_size
        self.steps_per_second = steps_per_second

    def update_position(self, dt):
        tx, ty = self.target_location
        cx, cy = self.cur_location

        dx = tx - cx
        dy = ty - cy
        if abs(dx) < 0.1:
            self.cur_location[0] = tx
            dx = 0
        if abs(dy) < 0.1:
            self.cur_location[1] = ty
            dy = 0
        if dx == 0 and dy == 0: return

        step_dist = self.speed * (dt / 1000.0) # dt in ms
        dist = (dx ** 2 + dy ** 2) ** 0.5
        ratio = min(1.0, step_dist / dist)
        self.cur_location[0] += dx * ratio
        self.cur_location[1] += dy * ratio

    def render(self, background):
        pass

class Agent(GameObject):

    def __init__(self, width, height, pix_square_size, steps_per_second):
        super().__init__(1, np.array([int(width / 2), height - 1], dtype=float), pix_square_size, steps_per_second)

    def get_cur_y(self):
        return self.target_location[1]

    def get_target_y(self):
        return self.target_location[1]

    def update_target(self, direction, map_width, map_height):
        delta = direction * (1 / self.steps_per_second)
        self.target_location = np.clip(self.cur_location + delta, [0, 0], [map_width - 1, map_height - 1])

    def render(self, background):
        agent_x = (self.cur_location[0] + 0.5) * self.pix_square_size
        agent_y = (self.cur_location[1] + 0.5) * self.pix_square_size
        agent_position = (agent_x, agent_y)
        pygame.draw.circle(
            background,
            (0, 0, 255),
            agent_position,
            self.pix_square_size / 3,
        )
        return agent_position

class Car(GameObject):

    def __init__(self, x, y, speed, grid_width, pix_square_size, steps_per_second, car_type_seed=0):
        super().__init__(speed * 0.5, np.array([x, y], dtype=float), pix_square_size, steps_per_second)
        self.car_type = (car_type_seed % 12)

        self.grid_width = grid_width
        self.width = pix_square_size
        self.height = pix_square_size

        self.object_id = id(self) % 1000
        object_image = pygame.image.load(f"sprites/cars/car-side-view{self.car_type}.png")
        self.image = pygame.transform.scale(object_image, (self.width, self.height))

        if speed < 0:
            self.image = pygame.transform.flip(self.image, flip_x=True, flip_y=False)

    def update_target(self, map_width):
        if self.speed > 0 and self.cur_location[0] >= map_width:
            self.cur_location[0] = 0
        if self.speed < 0 and self.cur_location[0] <= 0:
            self.cur_location[0] = map_width - 1
        self.target_location[0] = self.cur_location[0] + self.speed * (1 / self.steps_per_second)

    def render(self, background):
        background.blit(self.image, (self.cur_location[0] * self.pix_square_size, self.cur_location[1] * self.pix_square_size))
        self.update_target(self.grid_width)

    def __str__(self):
        return f"Car{self.object_id}: type={self.car_type}, cur_pos=({self.cur_location[0], self.cur_location[1]})"
