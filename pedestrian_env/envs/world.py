import pygame

from pedestrian_env.envs.road import Roads
from pedestrian_env.envs.game_object import Agent, Cars

class World:
    UP_REWARD_PER_UNIT = 5

    def __init__(self, agent_x_range, map_grid_width, map_grid_height, camera_width, pix_square_size, steps_per_second, random, debug, player_images, car_details_dict):
        self.random = random
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.pix_square_size = pix_square_size

        self.steps_per_second = steps_per_second

        self.agent = Agent(agent_x_range, map_grid_width, map_grid_height, pix_square_size, steps_per_second, player_images, debug)
        self.initial_player_y = self.agent.cur_location[1]
        self.roads = Roads(self.agent, camera_width, map_grid_height, self.random)
        self.cars = Cars.generate_cars(self.agent, self.roads, pix_square_size, map_grid_width, steps_per_second, car_details_dict, random)

        self.row_to_cars_dict = {}
        for car in self.cars.elements:
            for row in car.rows:
                if row not in self.row_to_cars_dict:
                    self.row_to_cars_dict[row] = []
                self.row_to_cars_dict[row].append(car)

        self.danger_map = [[True for _ in range(map_grid_width)] for _ in range(map_grid_height)]
        for y in self.roads.safe_row_idx_list:
            for x in range(map_grid_width):
                self.danger_map[y][x] = False # safe

        self.crosswalk_map = [[False for _ in range(map_grid_width)] for _ in range(map_grid_height)]
        for road in self.roads.elements:
            crosswalk = road.crosswalk
            if crosswalk is None: continue
            for y in range(crosswalk.top_row, crosswalk.end_row+2):
                for x in range(crosswalk.left_end, crosswalk.right_end+1):
                    self.crosswalk_map[y][x] = True # crosswalk

        self.reachable_map = [[True for _ in range(map_grid_width)] for _ in range(map_grid_height)]
        for y in range(map_grid_height):
            for x in range(map_grid_width):
                if agent_x_range[0] <= x <= agent_x_range[1]: continue
                self.reachable_map[y][x] = False # unreachable

        self.reward_y = [False for _ in range(map_grid_height)]
        self.reward_y[Agent.TARGET_LANE] = True
        self.reward_per_y = [0 for _ in range(map_grid_height)]
        for road in self.roads.elements:
            road_crossed_y = road.start_y - 1
            self.reward_y[road_crossed_y] = True
            cum_reward = int(self.steps_per_second * (self.initial_player_y - road_crossed_y)) * self.UP_REWARD_PER_UNIT
            self.reward_per_y[road_crossed_y] = cum_reward
        for y in range(map_grid_height-1, 0, -1):
            self.reward_per_y[y-1] = max(self.reward_per_y[y], self.reward_per_y[y-1])

    def target_lane_reached(self):
        _, agent_y = self.agent.get_cur_location_grid()
        return agent_y == Agent.TARGET_LANE

    def update_positions(self, dt):
        self.agent.update_position(dt)
        self.roads.activate_crosswalks()
        self.cars.update_positions(dt)

    def calculate_cum_crossing_rewards(self):
        _, agent_y = self.agent.get_cur_location_grid()
        return self.reward_per_y[agent_y]

    def render(self):
        map_width = self.map_grid_width * self.pix_square_size
        map_height = self.map_grid_height * self.pix_square_size
        canvas = pygame.Surface((map_width, map_height))
        self.roads.render(canvas, self.map_grid_width, self.pix_square_size)
        rendered_agent_position = self.agent.render(canvas)
        self.cars.render(canvas)
        return canvas, rendered_agent_position
