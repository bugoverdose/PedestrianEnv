import math

import pygame
import numpy as np

from pedestrian_env.envs.road import RowType
from pedestrian_env.envs.utils import is_overlapping

class GameObject:

    def __init__(self, cur_speed, cur_location, pix_square_size, steps_per_second):
        self.cur_speed = cur_speed
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

        step_dist = self.cur_speed * (dt / 1000.0) # dt in ms
        dist = (dx ** 2 + dy ** 2) ** 0.5
        ratio = min(1.0, step_dist / dist)
        self.cur_location[0] += dx * ratio
        self.cur_location[1] += dy * ratio

    def render(self, background):
        pass

class Agent(GameObject):
    TARGET_LANE = 0
    BODY_COLOR = (0, 0, 255)

    def __init__(self, map_grid_width, map_grid_height, pix_square_size, steps_per_second, debug):
        fixed_speed = 2
        if debug:
            fixed_speed = 10
        super().__init__(fixed_speed, np.array([int(map_grid_width / 2), map_grid_height - 1], dtype=float), pix_square_size, steps_per_second)
        self.radius = 1/3
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.is_dead = False
        self.MIN_X = 1 + Car.MAX_WIDTH
        self.MAX_X = self.map_grid_width - 1 - Car.MAX_WIDTH

    def get_cur_pos(self):
        x = self.cur_location[0]
        y = self.cur_location[1]
        return x, y, self.radius

    def get_cur_y(self):
        return self.cur_location[1]

    def get_target_y(self):
        return self.target_location[1]

    def update_target(self, direction):
        delta = direction * (self.cur_speed / self.steps_per_second)
        self.target_location = np.clip(self.cur_location + delta,
                                       [self.MIN_X, self.TARGET_LANE],
                                       [self.MAX_X, self.map_grid_height - 1])

    def set_dead(self):
        self.is_dead = True
        self.target_location = self.cur_location

    def render(self, background):
        agent_center_x = self.cur_location[0] * self.pix_square_size
        agent_center_y = self.cur_location[1] * self.pix_square_size
        agent_position = (agent_center_x, agent_center_y)
        if self.is_dead:
            radius_x = int(self.pix_square_size * self.radius * 1.4)
            radius_y = int(self.pix_square_size * self.radius * 0.4)
            flattened_rect = pygame.Rect(
                agent_center_x - radius_x,
                agent_center_y - radius_y,
                radius_x * 2,
                radius_y * 2
            )
            pygame.draw.ellipse(background, self.BODY_COLOR, flattened_rect)
        else:
            pygame.draw.circle(
                background,
                self.BODY_COLOR,
                agent_position,
                self.pix_square_size * self.radius,
            )
        return agent_position

class Car(GameObject):
    MAX_WIDTH = 4
    MAX_HEIGHT = 2
    CAR_SIZES = {1: [(1, 1.5), (1,2)], 2: [(2,3), (2,4)]} # key=height, value=(height, width)
    CAR_SPEEDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # NOTE: 0.4 is slightly faster than the player
    HEIGHT_BUFFER = 0.1
    BODY_COLOR = (255, 0, 0)

    def __init__(self, uid, initial_x, initial_y, car_width, car_height, go_right, road, map_grid_width, pix_square_size, steps_per_second, random):
        super().__init__(0, np.array([initial_x, initial_y], dtype=float), pix_square_size, steps_per_second)
        self.uid = uid
        self.map_grid_width = map_grid_width
        self.car_grid_width = car_width
        self.car_width = self.car_grid_width * pix_square_size
        self.car_height = car_height - 2 * self.HEIGHT_BUFFER
        self.car_height_pix = self.car_height * pix_square_size
        self.color = self.BODY_COLOR
        self.go_right = go_right
        self.random = random
        self.set_random_speed()
        self.road = road
        self.nearby_cars = [] # all the cars in the overlapping lane

    def get_cur_pos(self):
        left_x = self.cur_location[0] - (self.car_grid_width/2)
        right_x = self.cur_location[0] + (self.car_grid_width/2)
        top_y = self.cur_location[1] - (self.car_height/2)
        bottom_y = self.cur_location[1] + (self.car_height/2)
        return left_x, right_x, top_y, bottom_y

    def get_front_back_x(self):
        left_x = self.cur_location[0] - (self.car_grid_width/2)
        right_x = self.cur_location[0] + (self.car_grid_width/2)
        if self.go_right:
            return right_x, left_x
        return left_x, right_x

    def set_random_speed(self):
        self.cur_speed = self.random.choice(self.CAR_SPEEDS)
        if not self.go_right:
            self.cur_speed *= -1

    def restart_if_needed(self):
        # teleport to start of the lane when reaching the end
        if self.go_right and self.cur_location[0] >= self.map_grid_width:
            self.cur_location[0] = 0
            self.set_random_speed()
        if not self.go_right and self.cur_location[0] <= 0:
            self.cur_location[0] = self.map_grid_width - 1
            self.set_random_speed()

    def update_target(self):
        self.restart_if_needed()
        # stop in the current location when another car is in front of the car
        if len(self.nearby_cars) > 0:
            threshold = 0.1
            my_front, my_back = self.get_front_back_x()
            my_left, my_right, _, _ = self.get_cur_pos()
            for other_car in self.nearby_cars:
                other_front, other_back = other_car.get_front_back_x()
                # stop when another car is in front of me
                if self.go_right:
                    front_dist = other_back - my_front
                else:
                    front_dist = my_front - other_back
                if 0 < front_dist <= threshold:
                    self.target_location[0] = self.cur_location[0]
                    return

                # make the one with lower uid stop if the both cars are overlapping
                if self.uid > other_car.uid:
                    other_left, other_right, _, _ = other_car.get_cur_pos()
                    if is_overlapping(my_left, my_right, other_left, other_right):
                        self.target_location[0] = self.cur_location[0]
                        return

        # stop in front of the crosswalk when it is activated
        crosswalk = self.road.crosswalk
        if crosswalk is not None and crosswalk.is_active:
            car_left, car_right, _, _ = self.get_cur_pos()
            cw_left, cw_right = crosswalk.get_left_right()
            if is_overlapping(car_left, car_right, cw_left, cw_right):
                pass # overlapping. keep going to stop blocking the crosswalk
            else:
                car_front, _ = self.get_front_back_x()
                front_dist = min(abs(cw_left - car_front), abs(cw_right - car_front))
                if front_dist < crosswalk.threshold_distance:
                    self.target_location[0] = self.cur_location[0]
                    return
        self.target_location[0] = self.cur_location[0] + self.cur_speed * (1 / self.steps_per_second)

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
        if self.go_right:
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
        if self.go_right:
            window_x = (right_x * self.pix_square_size) - window_width - window_width*1
        else:
            window_x = (left_x * self.pix_square_size) + window_width*1
        window_y = (self.cur_location[1] * self.pix_square_size) - window_height/2
        pygame.draw.rect(background, (0, 0, 0), (window_x, window_y, window_width, window_height), border_radius=30)

    def __str__(self):
        return f"Car(cur_pos=({self.cur_location[0], self.cur_location[1]}), crosswalk=({self.crosswalk}))"

class Cars:
    def __init__(self, agent, elements):
        self.agent = agent
        self.elements = elements

    def update_positions(self, dt):
        for car in self.elements:
            car.update_target()
            car.update_position(dt)

    def render(self, background):
        for car in self.elements:
            car.render(background)

    def has_hit_agent(self):
        for car in self.elements:
            if self._check_collision(car):
                return True, car.road.penalty
        return False, 0

    def _check_collision(self, car):
        cx, cy, radius = self.agent.get_cur_pos() # circle
        left_x, right_x, top_y, bottom_y = car.get_cur_pos() # rectangle
        closest_x = max(left_x, min(cx, right_x))
        closest_y = max(top_y, min(cy, bottom_y))
        distance = math.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)
        return distance < radius

    @staticmethod
    def generate_cars(agent, row_types, max_height_dic, roads, pix_square_size, map_grid_width, map_grid_height, steps_per_second, random):
        cars = []
        uid = 0
        for row_idx in range(map_grid_height):
            if row_types[row_idx] == RowType.SAFE: continue
            initial_x = random.integers(0, map_grid_width)
            height = random.choice(list(range(1, max_height_dic[row_idx] + 1)))
            width = random.choice(Car.CAR_SIZES[height])[1]
            going_right = row_types[row_idx] == RowType.CAR_GOING_RIGHT
            row_idx += (height - 1) * 0.5
            cur_road = None
            for road in roads.elements:
                if road.row1 <= row_idx <= road.row2:
                    cur_road = road
                    break
            if cur_road is None:
                print("!!!", row_idx, roads.elements)
            uid += 1
            cars.append(Car(uid, initial_x, row_idx, width, height, going_right, cur_road, map_grid_width, pix_square_size, steps_per_second, random))
        for i in range(len(cars)):
            cur_car = cars[i]
            _, _, top_y1, bottom_y1 = cur_car.get_cur_pos()
            for j in range(len(cars)):
                if i == j: continue
                other_car = cars[j]
                other_car.get_cur_pos()
                _, _, top_y2, bottom_y2 = other_car.get_cur_pos()
                if max(top_y1, top_y2) <= min(bottom_y1, bottom_y2):
                    cur_car.nearby_cars.append(other_car) # overlapping lane
        return Cars(agent, cars)

    @staticmethod
    def _check_overlapping(car1, car2):
        l1, r1, t1, b1 = car1.get_cur_pos()
        l2, r2, t2, b2 = car2.get_cur_pos()
        if not is_overlapping(l1, r1, l2, r2):
            return False
        if not is_overlapping(t1, b1, t2, b2):
            return False
        return True

    def __str__(self):
        string = ""
        for car in self.elements:
            string += f"{car}\n"
        return string[:-1]
