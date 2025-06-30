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
        agent_center_x = (self.cur_location[0]) * self.pix_square_size
        agent_center_y = (self.cur_location[1]) * self.pix_square_size
        agent_position = (agent_center_x, agent_center_y)
        pygame.draw.circle(
            background,
            (0, 0, 255),
            agent_position,
            self.pix_square_size / 3,
        )
        return agent_position

class Car(GameObject):
    WIDTH = 4
    HEIGHT = 3 # NOTE: must be an odd number

    def __init__(self, initial_x, initial_y, speed, grid_width, pix_square_size, steps_per_second):
        super().__init__(speed * 0.5, np.array([initial_x, initial_y], dtype=float), pix_square_size, steps_per_second)
        self.grid_width = grid_width
        self.width = self.WIDTH * pix_square_size
        self.height = self.HEIGHT * pix_square_size
        self.color = (255, 0, 0)

    def update_target(self, map_width):
        if self.speed > 0 and self.cur_location[0] >= map_width:
            self.cur_location[0] = 0
        if self.speed < 0 and self.cur_location[0] <= 0:
            self.cur_location[0] = map_width - 1
        self.target_location[0] = self.cur_location[0] + self.speed * (1 / self.steps_per_second)

    def render(self, background):
        left_x = self.cur_location[0] - (self.WIDTH/2)
        top_y = self.cur_location[1] - (self.HEIGHT/2)
        car_rect = pygame.Rect(left_x * self.pix_square_size, top_y * self.pix_square_size,
                               self.width, self.height)
        pygame.draw.rect(background, self.color, car_rect)
        self.update_target(self.grid_width)

    def __str__(self):
        return f"Car(cur_pos=({self.cur_location[0], self.cur_location[1]})"
