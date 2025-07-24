import pygame

from pedestrian_env.envs.road import Roads
from pedestrian_env.envs.game_object import Agent, Cars
from pedestrian_env.envs.grid_type import GridType

class World:
    UP_REWARD_PER_UNIT = 5

    def __init__(self, agent_x_range, map_grid_width, map_grid_height, camera_size, pix_square_size, steps_per_second, random, debug, render_sprite):
        self.random = random
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.pix_square_size = pix_square_size

        self.steps_per_second = steps_per_second

        self.agent = Agent(agent_x_range, map_grid_width, map_grid_height, pix_square_size, steps_per_second, debug)
        self.initial_player_y = self.agent.cur_location[1]

        self.roads = Roads(self.agent, camera_size, map_grid_height, self.random)
        self.cars = Cars.generate_cars(self.agent, self.roads, pix_square_size, map_grid_width, steps_per_second, random, render_sprite)

        self.row_to_cars_dict = {}
        for car in self.cars.elements:
            for row in car.rows:
                if row not in self.row_to_cars_dict:
                    self.row_to_cars_dict[row] = []
                self.row_to_cars_dict[row].append(car)
            
        self.grid_type_map = [[GridType.DANGER for _ in range(map_grid_width)] for _ in range(map_grid_height)]
        for y in self.roads.safe_row_idx_list:
            for x in range(map_grid_width):
                self.grid_type_map[y][x] = GridType.SAFE
        for road in self.roads.elements:
            crosswalk = road.crosswalk
            if crosswalk is None: continue
            for y in range(crosswalk.top_row, crosswalk.end_row+2):
                for x in range(crosswalk.left_end, crosswalk.right_end+1):
                    self.grid_type_map[y][x] = GridType.CROSSWALK
        for y in range(map_grid_height):
            for x in range(map_grid_width):
                if agent_x_range[0] <= x <= agent_x_range[1]: continue
                self.grid_type_map[y][x] = GridType.UNREACHABLE

    def target_lane_reached(self):
        cur_y = self.agent.cur_location[1]
        return cur_y == Agent.TARGET_LANE

    def update_positions(self, dt):
        self.agent.update_position(dt)
        self.roads.activate_crosswalks()
        self.cars.update_positions(dt)

    def calculate_up_rewards(self):
        return int(self.steps_per_second * (self.initial_player_y - self.agent.get_target_y())) * self.UP_REWARD_PER_UNIT

    def render(self):
        map_width = self.map_grid_width * self.pix_square_size
        map_height = self.map_grid_height * self.pix_square_size
        canvas = pygame.Surface((map_width, map_height))
        self.roads.render(canvas, self.map_grid_width, self.pix_square_size)
        rendered_agent_position = self.agent.render(canvas)
        self.cars.render(canvas)
        return canvas, rendered_agent_position
