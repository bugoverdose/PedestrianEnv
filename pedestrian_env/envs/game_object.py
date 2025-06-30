import pygame
import numpy as np

class Agent:

    def __init__(self, width, height, pix_square_size):
        self.cur_location = np.array([int(width / 2), height - 1], dtype=float)
        self.target_location = self.cur_location
        self.pix_square_size = pix_square_size

    def get_cur_y(self):
        return self.cur_location[1]

    def update_position(self, dt):
        agent_speed = 1.0
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

        step_dist = agent_speed * (dt / 1000.0) # dt in ms
        dist = (dx ** 2 + dy ** 2) ** 0.5
        ratio = min(1.0, step_dist / dist)
        self.cur_location[0] += dx * ratio
        self.cur_location[1] += dy * ratio

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

class GameObject:

    def __init__(self, image_path, x, y, width, height, pix_square_size):
        self.object_id = id(self) % 1000
        object_image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(object_image, (width, height))

        self.x = x
        self.y = y

        self.width = width
        self.height = height
        self.pix_square_size = pix_square_size

    def render(self, background):
        pass

class Car(GameObject):

    def __init__(self, x, y, speed, pix_square_size, car_type_seed = 0):
        self.car_type = (car_type_seed % 12)
        self.speed = speed
        super().__init__(f"sprites/cars/car-side-view{self.car_type}.png", x, y, pix_square_size, pix_square_size, pix_square_size)
        if speed < 0:
            self.image = pygame.transform.flip(self.image, flip_x=True, flip_y=False)

    def move(self, max_width):
        self.x += self.speed
        if self.speed > 0 and self.x >= max_width:
            self.x = 0
        if self.speed < 0 and self.x <= 0:
            self.x = max_width - 1

    def __str__(self):
        return f"Car{self.object_id}: type={self.car_type}, cur_pos=({self.x, self.y})"

    def render(self, background):
        background.blit(self.image, (self.x * self.pix_square_size, self.y * self.pix_square_size))
