import os

import pygame
import numpy as np

from pedestrian_env.envs.action import Action, ACTION_TO_DELTA, ACTION_DURATION
from pedestrian_env.envs.utils import is_overlapping
from pedestrian_env.envs.road import Road, CrossWalk
from pedestrian_env.envs.car_details import CAR_CANDIDATES, CARS_PER_LANE_PAIR_COMPOSITION

AGENT_STATUS_ALIVE = "alive"
AGENT_STATUS_DEAD = "dead"

class GameObject:

    def __init__(self, default_speed, cur_location, pix_square_size, steps_per_second):
        self.default_speed = default_speed
        self.cur_location = cur_location
        self.target_location = self.cur_location
        self.pix_square_size = pix_square_size
        self.steps_per_second = steps_per_second
        self.is_moving = False

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

        if dx != 0 and dy != 0: # diagonal move
            dist = (dx ** 2 + dy ** 2) ** 0.5
        else: # straight move
            dist = max(abs(dx), abs(dy))
        step_dist = self.default_speed * (dt / 1000.0) # dt in ms
        ratio = min(1.0, step_dist / dist)
        self.cur_location[0] += dx * ratio
        self.cur_location[1] += dy * ratio
        self.is_moving = True

    def stop(self):
        self.target_location = self.cur_location
        self.is_moving = False

    def render(self, background):
        pass

class Agent(GameObject):
    TARGET_LANE = 0
    BODY_COLOR = (0, 50, 255) # (0, 0, 255)
    EYE_COLOR = (0, 0, 0)
    RADIUS = 0.25

    def __init__(self, agent_x_range, map_grid_width, map_grid_height, pix_square_size, steps_per_second, player_asset_info, debug):
        fixed_speed = 2.5
        if debug:
            fixed_speed *= 5
        self.init_pos = (int(map_grid_width / 2), map_grid_height - 1)
        cur_location = np.array([self.init_pos[0], self.init_pos[1]], dtype=float)
        super().__init__(fixed_speed, cur_location, pix_square_size, steps_per_second)
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.is_dead = False
        self.MIN_X = agent_x_range[0]
        self.MAX_X = agent_x_range[1]
        self.last_direction = Action.DOWN.value
        self.images = player_asset_info["images"]
        self.sizes = player_asset_info["sizes"]
        self.mini_step_count = 0

    def get_cur_grid_pos(self):
        center_x = self.cur_location[0]
        center_y = self.cur_location[1]
        sub_dir = AGENT_STATUS_DEAD if self.is_dead else AGENT_STATUS_ALIVE
        direction = self.mini_step_count % ACTION_DURATION[self.last_direction]
        [player_width, player_height] = self.sizes[self.last_direction][sub_dir][direction]
        top_y = center_y - player_height/2
        bottom_y = top_y + player_height
        left_x = center_x - player_width/2
        right_x = left_x + player_width
        return [top_y, bottom_y, left_x, right_x]

    def get_cur_location_grid(self):
        return int(round(self.cur_location[0])), int(round(self.cur_location[1]))

    def get_cur_location_rounded(self):
        return round(self.cur_location[0], 2), round(self.cur_location[1], 2)

    def get_target_y(self):
        return round(self.target_location[1], 2)

    def update_target(self, action_value):
        if action_value != Action.NOTHING.value:
            self.last_direction = action_value
        delta = ACTION_TO_DELTA[action_value] * (self.default_speed / self.steps_per_second) * ACTION_DURATION[action_value]
        self.target_location = np.clip(self.cur_location + delta,
                                       [self.MIN_X, self.TARGET_LANE],
                                       [self.MAX_X, self.map_grid_height - 1])

    def set_dead(self):
        self.is_dead = True
        self.stop()

    def render(self, background):
        adjustment = 0.5 * self.pix_square_size
        agent_center_x = self.cur_location[0] * self.pix_square_size
        agent_center_y = self.cur_location[1] * self.pix_square_size + adjustment
        agent_position = (agent_center_x, agent_center_y)
        if self.images is not None:
            sub_dir = AGENT_STATUS_DEAD if self.is_dead else AGENT_STATUS_ALIVE
            direction = self.mini_step_count % ACTION_DURATION[self.last_direction]
            image = self.images[self.last_direction][sub_dir][direction]
            [agent_grid_width, agent_grid_height] = self.sizes[self.last_direction][sub_dir][direction]
            left_x_pix = agent_center_x - (agent_grid_width * self.pix_square_size / 2)
            top_y_pix = agent_center_y - (agent_grid_height * self.pix_square_size / 2)
            background.blit(image, (left_x_pix, top_y_pix))
        return agent_position

class Car(GameObject):
    CAR_SPEEDS = [3.0, 3.5, 4.0, 4.5]  # NOTE: 2 is slightly faster than the player

    def __init__(self, uid, initial_x, start_row_idx, car_grid_height, road, map_grid_width, pix_square_size, steps_per_second, car_details, random):
        row_indices = []
        for i in range(car_grid_height):
            row_indices.append(start_row_idx + i)
        initial_y = np.mean(row_indices)
        cur_location = np.array([initial_x, initial_y], dtype=float)
        super().__init__(0, cur_location, pix_square_size, steps_per_second)
        self.uid = uid
        self.rows = row_indices
        self.map_grid_width = map_grid_width
        self.car_details = car_details
        self.random = random
        self.set_random_speed()
        self.road = road
        self.nearby_cars = [] # all the cars in the overlapping lane

    def get_cur_speed(self):
        return self.default_speed if self.is_moving else 0

    def get_cur_x_pos(self):
        left_x = self.cur_location[0] - (self.car_details.car_grid_width/2)
        right_x = self.cur_location[0] + (self.car_details.car_grid_width/2)
        return left_x, right_x

    def get_cur_y_pos(self):
        top_y = self.cur_location[1] - (self.car_details.car_grid_height/2)
        bottom_y = self.cur_location[1] + (self.car_details.car_grid_height/2)
        return top_y, bottom_y

    def set_random_speed(self):
        self.default_speed = self.random.choice(self.CAR_SPEEDS)
        if not self.car_details.go_right:
            self.default_speed *= -1

    def restart_if_needed(self):
        # teleport to start of the lane when reaching the end
        if self.car_details.go_right and self.cur_location[0] >= self.map_grid_width:
            self.cur_location[0] = 0
            self.set_random_speed()
        if not self.car_details.go_right and self.cur_location[0] <= 0:
            self.cur_location[0] = self.map_grid_width - 1
            self.set_random_speed()

    def update_target(self, dt):
        self.restart_if_needed()
        # stop in the current location when another car is in front of the car
        if len(self.nearby_cars) > 0:
            threshold = 0.1
            my_left, my_right = self.get_cur_x_pos()
            for other_car in self.nearby_cars:
                other_left, other_right = other_car.get_cur_x_pos()
                # stop when another car is in front of me
                if self.car_details.go_right:
                    front_dist = other_left - my_right
                else:
                    front_dist = my_left - other_right
                if 0 < front_dist <= threshold:
                    self.stop()
                    return

                # make the one in the back stop if the both cars are overlapping
                # make the one with lower uid stop if the front positions are the same
                if self.car_details.go_right:
                    front_diff = other_right - my_right
                else:
                    front_diff = my_left - other_left
                if front_diff > 0 or (front_diff == 0 and self.uid < other_car.uid):
                    if is_overlapping(my_left, my_right, other_left, other_right):
                        self.stop()
                        return

        # stop in front of the crosswalk when it is activated
        crosswalk = self.road.crosswalk
        if crosswalk is not None and crosswalk.is_active:
            car_left, car_right = self.get_cur_x_pos()
            cw_left, cw_right = crosswalk.get_activation_left_right()
            if self.car_details.go_right:
                front_dist = abs(cw_left - car_right)
            else:
                front_dist = abs(cw_right - car_left)
            if front_dist < CrossWalk.THRESHOLD_DISTANCE:
                if is_overlapping(car_left, car_right, cw_left, cw_right):
                    pass # keep going to clear up the crosswalk
                else:
                    self.stop()
                    return
        self.target_location[0] = self.cur_location[0] + (self.default_speed * dt / 1000)
        self.is_moving = True # moving unless stopped

    def render(self, background):
        left_x, _ = self.get_cur_x_pos()
        top_y, _ = self.get_cur_y_pos()
        left_x_pix = left_x * self.pix_square_size
        top_y_pix = top_y * self.pix_square_size
        adjustment = 0.5 * self.pix_square_size
        if self.car_details.image is not None:
            background.blit(self.car_details.image, (left_x_pix, top_y_pix + adjustment))

    def __str__(self):
        return f"Car(cur_pos=({self.cur_location[0], self.cur_location[1]}), crosswalk=({self.road.crosswalk}))"

class Cars:
    def __init__(self, agent, elements):
        self.agent = agent
        self.elements = elements

    def update_positions(self, dt):
        for car in self.elements:
            car.update_target(dt)
            car.update_position(dt)

    def render(self, background):
        for car in self.elements:
            car.render(background)

    @staticmethod
    def generate_cars(agent, roads, pix_square_size, map_grid_width, steps_per_second, car_details_dict, random):
        cars = []
        uid = 0
        for road in roads.elements:
            road_height = road.end_y - road.start_y + 1
            for i in range(0, road_height, Road.COMPOSITION_SIZE):
                going_right = road.going_right[i]
                cars_per_lane_pair = random.choice(CARS_PER_LANE_PAIR_COMPOSITION)
                for j in cars_per_lane_pair.keys():
                    start_row_idx = road.start_y + i + j
                    for height in cars_per_lane_pair[j]:
                        uid += 1
                        (car_name, _) = random.choice(CAR_CANDIDATES[height])
                        initial_x = random.integers(0, map_grid_width)
                        car_details = car_details_dict[car_name][road.car_color_type][going_right]
                        cars.append(Car(uid, initial_x, start_row_idx, height, road, map_grid_width, pix_square_size, steps_per_second, car_details, random))

        for i in range(len(cars)):
            cur_car = cars[i]
            top_y1, bottom_y1 = cur_car.get_cur_y_pos()
            for j in range(len(cars)):
                if i == j: continue
                other_car = cars[j]
                top_y2, bottom_y2 = other_car.get_cur_y_pos()
                if max(top_y1, top_y2) <= min(bottom_y1, bottom_y2):
                    cur_car.nearby_cars.append(other_car) # overlapping lane
        return Cars(agent, cars)

    @staticmethod
    def _check_overlapping(car1, car2):
        l1, r1 = car1.get_cur_x_pos()
        t1, b1 = car1.get_cur_y_pos()
        l2, r2 = car2.get_cur_x_pos()
        t2, b2 = car2.get_cur_y_pos()
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

def load_player_asset_info(pix_square_size, render_sprite):
    directions = [Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]
    size_ratio = {
        "player_UP_0_cropped.png": [57,73],
        "player_UP_1_cropped.png": [53,69],
        "player_UP_2_cropped.png": [57,73],
        "player_UP_3_cropped.png": [53,69],

        "player_DOWN_0_cropped.png": [57,73],
        "player_DOWN_1_cropped.png": [53,69],
        "player_DOWN_2_cropped.png": [57,73],
        "player_DOWN_3_cropped.png": [53,69],

        "player_RIGHT_0_cropped.png": [49,69],
        "player_RIGHT_1_cropped.png": [49,65],
        "player_RIGHT_2_cropped.png": [49,69],
        "player_RIGHT_3_cropped.png": [49,65],

        "player_LEFT_0_cropped.png": [49,69],
        "player_LEFT_1_cropped.png": [49,65],
        "player_LEFT_2_cropped.png": [49,69],
        "player_LEFT_3_cropped.png": [49,65],
    }
    images = {} if render_sprite else None
    sizes = {}
    for d in directions:
        if render_sprite:
            images[d.value] = {}
            images[d.value][AGENT_STATUS_ALIVE] = []
            images[d.value][AGENT_STATUS_DEAD] = []
        sizes[d.value] = {}
        sizes[d.value][AGENT_STATUS_ALIVE] = []
        sizes[d.value][AGENT_STATUS_DEAD] = []
        for i in range(4):
            for status in [AGENT_STATUS_ALIVE, AGENT_STATUS_DEAD]:
                file_name = f"player_{str(d)}_{i}_cropped.png"
                current_dir = os.path.dirname(os.path.abspath(__file__))
                image_path = os.path.join(current_dir, "..", f"sprites/player_color/", file_name)
                [width_ratio, height_ratio] = size_ratio[file_name]
                agent_grid_height = (height_ratio / 140) * (0.4 if status == AGENT_STATUS_DEAD else 1)
                agent_grid_width = (width_ratio / 140) * (2.4 if status == AGENT_STATUS_DEAD else 1)
                if render_sprite:
                    object_image = pygame.image.load(image_path)
                    image = pygame.transform.scale(object_image, (agent_grid_width * pix_square_size, agent_grid_height * pix_square_size))
                    images[d.value][status].append(image)
                sizes[d.value][status].append([agent_grid_width, agent_grid_height])
    return {"images": images, "sizes": sizes}
