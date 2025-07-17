import pygame

from pedestrian_env.envs.road import Roads
from pedestrian_env.envs.game_object import Agent, Cars

class World:
    UP_REWARD_PER_UNIT = 5

    def __init__(self, map_grid_width, map_grid_height, camera_size, pix_square_size, steps_per_second, random, debug):
        self.random = random
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.pix_square_size = pix_square_size

        self.steps_per_second = steps_per_second
        self.agent = Agent(map_grid_width, map_grid_height, pix_square_size, steps_per_second, debug)
        self.initial_player_y = self.agent.get_cur_y()

        self.roads = Roads(self.agent, camera_size, map_grid_height, self.random)
        self.cars = Cars.generate_cars(self.agent, self.roads, pix_square_size, map_grid_width, steps_per_second, random, render_sprite=True)

    def target_lane_reached(self):
        cur_y = self.agent.get_cur_y()
        return cur_y == Agent.TARGET_LANE

    def update_positions(self, dt):
        self.agent.update_position(dt)
        self.roads.update_crosswalk_activation()
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
