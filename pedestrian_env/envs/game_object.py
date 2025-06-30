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
    TARGET_LANE = 0

    def __init__(self, map_grid_width, map_grid_height, pix_square_size, steps_per_second, debug):
        speed = 2
        if debug:
            speed = 10
        super().__init__(speed, np.array([int(map_grid_width / 2), map_grid_height - 1], dtype=float), pix_square_size, steps_per_second)
        self.radius = 1/3
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height

    def get_cur_pos(self):
        x = self.cur_location[0]
        y = self.cur_location[1]
        return x, y, self.radius

    def get_cur_y(self):
        return self.cur_location[1]

    def get_target_y(self):
        return self.target_location[1]

    def update_target(self, direction):
        delta = direction * (self.speed / self.steps_per_second)
        self.target_location = np.clip(self.cur_location + delta,
                                       [1 + Car.MAX_WIDTH, self.TARGET_LANE],
                                       [self.map_grid_width - 1 - Car.MAX_WIDTH, self.map_grid_height - 1])

    def render(self, background):
        agent_center_x = self.cur_location[0] * self.pix_square_size
        agent_center_y = self.cur_location[1] * self.pix_square_size
        agent_position = (agent_center_x, agent_center_y)
        pygame.draw.circle(
            background,
            (0, 0, 255),
            agent_position,
            self.pix_square_size * self.radius,
        )
        return agent_position

class Car(GameObject):
    MAX_WIDTH = 4
    MAX_HEIGHT = 2
    CAR_SIZES = {1: [(1, 1.5), (1,2)], 2: [(2, 2.5), (2,3), (2, 3.5), (2,4)]} # key=height, value=(height, width)
    HEIGHT_BUFFER = 0.1

    def __init__(self, initial_x, initial_y, car_width, car_height, speed, map_grid_width, pix_square_size, steps_per_second):
        super().__init__(speed * 0.5, np.array([initial_x, initial_y], dtype=float), pix_square_size, steps_per_second)
        self.map_grid_width = map_grid_width
        self.car_grid_width = car_width
        self.car_width = self.car_grid_width * pix_square_size
        self.car_height = car_height - 2 * self.HEIGHT_BUFFER
        self.car_height_pix = self.car_height * pix_square_size
        self.color = (255, 0, 0)

    def get_cur_pos(self):
        left_x = self.cur_location[0] - (self.car_grid_width/2)
        right_x = self.cur_location[0] + (self.car_grid_width/2)
        top_y = self.cur_location[1] - (self.car_height/2)
        bottom_y = self.cur_location[1] + (self.car_height/2)
        return left_x, right_x, top_y, bottom_y

    def update_target(self):
        if self.speed > 0 and self.cur_location[0] >= self.map_grid_width:
            self.cur_location[0] = 0
        if self.speed < 0 and self.cur_location[0] <= 0:
            self.cur_location[0] = self.map_grid_width - 1
        self.target_location[0] = self.cur_location[0] + self.speed * (1 / self.steps_per_second)

    def render(self, background):
        left_x, right_x, top_y, bottom_y = self.get_cur_pos()
        center_x_pix = self.cur_location[0] * self.pix_square_size
        center_y_pix = self.cur_location[1] * self.pix_square_size
        left_x_pix = left_x * self.pix_square_size
        right_x_pix = right_x * self.pix_square_size
        top_y_pix = top_y * self.pix_square_size

        # draw car
        radius = self.car_height_pix/2
        rect_width = self.car_width * 1.05 - radius
        if self.speed > 0:
            car_rect = pygame.Rect(left_x_pix, top_y_pix, rect_width, self.car_height_pix)
            pygame.draw.rect(background, self.color, car_rect, border_radius=10)
            pygame.draw.circle(background, self.color, [center_x_pix + (self.car_width * 0.5 - radius), center_y_pix], self.car_height_pix / 2)
        else:
            car_rect = pygame.Rect(right_x_pix-rect_width, top_y_pix, rect_width, self.car_height_pix)
            pygame.draw.rect(background, self.color, car_rect, border_radius=10)
            pygame.draw.circle(background, self.color, [center_x_pix - (self.car_width * 0.5 - radius), center_y_pix], self.car_height_pix / 2)

        # draw window
        window_width = self.car_height_pix/3
        window_height = (self.car_height_pix * 0.7)
        if self.speed > 0:
            window_x = (right_x * self.pix_square_size) - window_width - window_width*1
        else:
            window_x = (left_x * self.pix_square_size) + window_width*1
        window_y = (self.cur_location[1] * self.pix_square_size) - window_height/2
        pygame.draw.rect(background, (0, 0, 0), (window_x, window_y, window_width, window_height), border_radius=30)

        self.update_target()

    def __str__(self):
        return f"Car(cur_pos=({self.cur_location[0], self.cur_location[1]})"
