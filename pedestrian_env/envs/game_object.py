import pygame
import numpy as np

from pedestrian_env.envs.action import Action, ACTION_TO_DELTA
from pedestrian_env.envs.utils import is_overlapping, is_overlapping_circle_and_rectangle
from pedestrian_env.envs.road import Road, CrossWalk
from pedestrian_env.envs.car_details import CarDetail, CAR_CANDIDATES, CARS_PER_LANE_PAIR_COMPOSITION

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

    def __init__(self, agent_x_range, map_grid_width, map_grid_height, pix_square_size, steps_per_second, debug):
        fixed_speed = 2
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
        self.last_direction = ACTION_TO_DELTA[Action.UP.value]

    def get_cur_location_grid(self):
        return int(round(self.cur_location[0])), int(round(self.cur_location[1]))

    def get_cur_location_rounded(self):
        return round(self.cur_location[0], 2), round(self.cur_location[1], 2)

    def get_target_y(self):
        return round(self.target_location[1], 2)

    def update_target(self, action_value):
        direction = ACTION_TO_DELTA[action_value]
        if action_value != Action.NOTHING.value:
            self.last_direction = direction
        delta = direction * (self.default_speed / self.steps_per_second)
        self.target_location = np.clip(self.cur_location + delta,
                                       [self.MIN_X, self.TARGET_LANE],
                                       [self.MAX_X, self.map_grid_height - 1])

    def set_dead(self):
        self.is_dead = True
        self.stop()

    def render(self, background):
        agent_center_x = self.cur_location[0] * self.pix_square_size
        agent_center_y = self.cur_location[1] * self.pix_square_size
        agent_position = (agent_center_x, agent_center_y)
        if self.is_dead:
            radius_x = int(self.pix_square_size * self.RADIUS * 1.4)
            radius_y = int(self.pix_square_size * self.RADIUS * 0.4)
            flattened_rect = pygame.Rect(
                agent_center_x - radius_x,
                agent_center_y - radius_y,
                radius_x * 2,
                radius_y * 2
            )
            pygame.draw.ellipse(background, self.BODY_COLOR, flattened_rect)
        else:
            # draw body
            radius_pix = self.pix_square_size * self.RADIUS
            pygame.draw.circle(
                background,
                self.BODY_COLOR,
                agent_position,
                radius_pix,
            )

            # draw eys
            [dx, dy] = self.last_direction
            px, py = -dy, dx # for rotation
            eyes_apart = radius_pix * 0.3
            eyes_from_head = radius_pix * 0.6

            eye_width, eye_height = radius_pix * 0.15, radius_pix * 0.4
            if dx != 0: # left or right
                eye_width, eye_height = eye_height, eye_width
            left_eye_rect = pygame.Rect(
                (agent_center_x + dx * eyes_from_head + px * eyes_apart) - (eye_width // 2),
                (agent_center_y + dy * eyes_from_head + py * eyes_apart) - (eye_height // 2),
                eye_width,
                eye_height,
            )
            right_eye_rect = pygame.Rect(
                (agent_center_x + dx * eyes_from_head - px * eyes_apart) - (eye_width // 2),
                (agent_center_y + dy * eyes_from_head - py * eyes_apart)- (eye_height // 2),
                eye_width,
                eye_height,
            )
            pygame.draw.ellipse(background, self.EYE_COLOR, left_eye_rect)
            pygame.draw.ellipse(background, self.EYE_COLOR, right_eye_rect)

        return agent_position

class Car(GameObject):
    CAR_SPEEDS = [3.0, 3.5, 4.0, 4.5]  # NOTE: 2 is slightly faster than the player
    HEIGHT_BUFFER = 0.1 # NOTE: needed to handle the overlap between the lanes

    def __init__(self, uid, car_name, initial_x, start_row_idx, car_grid_height, size_ratio, go_right, road, map_grid_width, pix_square_size, steps_per_second, random, render_sprite):
        row_indices = []
        for i in range(car_grid_height):
            row_indices.append(start_row_idx + i)
        initial_y = np.mean(row_indices)
        cur_location = np.array([initial_x, initial_y], dtype=float)
        super().__init__(0, cur_location, pix_square_size, steps_per_second)
        self.uid = uid
        self.rows = row_indices
        self.map_grid_width = map_grid_width
        self.car_grid_width = car_grid_height * float(size_ratio)
        self.car_width = self.car_grid_width * pix_square_size
        self.car_grid_height = car_grid_height - self.HEIGHT_BUFFER
        self.car_height = self.car_grid_height * pix_square_size
        self.row_indices = row_indices
        self.car_detail = CarDetail(car_name, road.car_color_type, go_right)
        self.go_right = go_right
        self.random = random
        self.set_random_speed()
        self.road = road
        self.nearby_cars = [] # all the cars in the overlapping lane
        self.image = None
        if render_sprite:
            object_image = pygame.image.load(self.car_detail.image_path)
            self.image = pygame.transform.scale(object_image, (self.car_width, self.car_height))

    def get_cur_speed(self):
        return self.default_speed if self.is_moving else 0

    def get_cur_x_pos(self):
        left_x = self.cur_location[0] - (self.car_grid_width/2)
        right_x = self.cur_location[0] + (self.car_grid_width/2)
        return left_x, right_x

    def get_cur_y_pos(self):
        top_y = self.cur_location[1] - (self.car_grid_height/2)
        bottom_y = self.cur_location[1] + (self.car_grid_height/2)
        return top_y, bottom_y

    def set_random_speed(self):
        self.default_speed = self.random.choice(self.CAR_SPEEDS)
        if not self.go_right:
            self.default_speed *= -1

    def restart_if_needed(self):
        # teleport to start of the lane when reaching the end
        if self.go_right and self.cur_location[0] >= self.map_grid_width:
            self.cur_location[0] = 0
            self.set_random_speed()
        if not self.go_right and self.cur_location[0] <= 0:
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
                if self.go_right:
                    front_dist = other_left - my_right
                else:
                    front_dist = my_left - other_right
                if 0 < front_dist <= threshold:
                    self.stop()
                    return

                # make the one in the back stop if the both cars are overlapping
                # make the one with lower uid stop if the front positions are the same
                if self.go_right:
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
            if self.go_right:
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
        left_x, right_x = self.get_cur_x_pos()
        top_y, bottom_y = self.get_cur_y_pos()
        left_x_pix = left_x * self.pix_square_size
        top_y_pix = top_y * self.pix_square_size
        # render image or rectangle
        if self.image is not None:
            background.blit(self.image, (left_x_pix, top_y_pix))
        else:
            center_x_pix = self.cur_location[0] * self.pix_square_size
            center_y_pix = self.cur_location[1] * self.pix_square_size
            right_x_pix = right_x * self.pix_square_size

            # draw car
            radius = self.car_height/2
            rect_width = self.car_width * 1.05 - radius
            if self.go_right:
                car_rect = pygame.Rect(left_x_pix, top_y_pix, rect_width, self.car_height)
                pygame.draw.rect(background, self.car_detail.color, car_rect, border_radius=10)
                pygame.draw.circle(background, self.car_detail.color, [center_x_pix + (self.car_width * 0.5 - radius), center_y_pix], self.car_height / 2)
            else:
                car_rect = pygame.Rect(right_x_pix-rect_width, top_y_pix, rect_width, self.car_height)
                pygame.draw.rect(background, self.car_detail.color, car_rect, border_radius=10)
                pygame.draw.circle(background, self.car_detail.color, [center_x_pix - (self.car_width * 0.5 - radius), center_y_pix], self.car_height / 2)

            # draw window
            window_width = self.car_height/3
            window_height = (self.car_height * 0.7)
            if self.go_right:
                window_x = (right_x * self.pix_square_size) - window_width - window_width*1
            else:
                window_x = (left_x * self.pix_square_size) + window_width*1
            window_y = (self.cur_location[1] * self.pix_square_size) - window_height/2
            pygame.draw.rect(background, (0, 0, 0), (window_x, window_y, window_width, window_height), border_radius=30)

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

    def has_hit_agent(self):
        for car in self.elements:
            if self._check_collision(car):
                return True, car.car_detail.penalty
        return False, 0

    def _check_collision(self, car):
        [cx, cy] = self.agent.cur_location
        left_x, right_x = car.get_cur_x_pos()
        top_y, bottom_y = car.get_cur_y_pos()
        return is_overlapping_circle_and_rectangle((cx, cy, Agent.RADIUS), (left_x, right_x, top_y, bottom_y))

    @staticmethod
    def generate_cars(agent, roads, pix_square_size, map_grid_width, steps_per_second, random, render_sprite=False):
        cars = []
        uid = 0
        for road in roads.elements:
            road_height = road.end_y - road.start_y + 1
            for i in range(0, road_height, Road.COMPOSITION_SIZE):
                cars_per_lane_pair = random.choice(CARS_PER_LANE_PAIR_COMPOSITION)
                for j in cars_per_lane_pair.keys():
                    start_row_idx = road.start_y + i + j
                    for height in cars_per_lane_pair[j]:
                        uid += 1
                        (car_name, ratio) = random.choice(CAR_CANDIDATES[height])
                        initial_x = random.integers(0, map_grid_width)
                        going_right = road.going_right[i]
                        cars.append(Car(uid, car_name, initial_x, start_row_idx, height, ratio, going_right, road, map_grid_width, pix_square_size, steps_per_second, random, render_sprite))

        # NOTE(maximum difficulty): add one height=1 car for each lane & add one height=2 car for each 2 lanes
        # for road in roads.elements:
        #     for height in CAR_CANDIDATES.keys():
        #         for i in range(0, road.end_y - road.start_y + 1, height):
        #             uid += 1
        #             row_idx = road.start_y + i
        #             height = min(roads.max_height_dic[row_idx], height)
        #             (car_name, ratio) = random.choice(CAR_CANDIDATES[height])
        #             initial_x = random.integers(0, map_grid_width)
        #             going_right = road.going_right[i]
        #             cars.append(Car(uid, car_name, initial_x, row_idx, height, ratio, going_right, road, map_grid_width, pix_square_size, steps_per_second, random, render_sprite))

        # add nearby_cars info
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
