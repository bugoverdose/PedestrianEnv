import pygame

from pedestrian_env.envs.road import Roads
from pedestrian_env.envs.game_object import Agent, Cars
from pedestrian_env.envs.car_details import get_max_car_grid_width

class World:
    UP_REWARD_PER_UNIT = 5

    def __init__(self, agent_x_range, map_grid_width, map_grid_height, camera_size, pix_square_size, steps_per_second, random, debug, render_sprite):
        self.random = random
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.pix_square_size = pix_square_size

        self.steps_per_second = steps_per_second

        self.agent = Agent(agent_x_range, map_grid_width, map_grid_height, pix_square_size, steps_per_second, debug)
        self.initial_player_y = self.agent.get_cur_y()

        self.roads = Roads(self.agent, camera_size, map_grid_height, self.random)
        self.cars = Cars.generate_cars(self.agent, self.roads, pix_square_size, map_grid_width, steps_per_second, random, render_sprite)

    def target_lane_reached(self):
        cur_y = self.agent.get_cur_y()
        return cur_y == Agent.TARGET_LANE

    def dist_until_roads(self, max_dist = 1):
        [_, agent_y] = self.agent.cur_location
        agent_up_y, agent_down_y = agent_y - Agent.RADIUS, agent_y + Agent.RADIUS
        closest_front_dist, closest_back_dist = max_dist, max_dist
        for road in self.roads.elements:
            road_up_y, road_down_y = road.row1 - 0.5, road.row2 + 0.5 # row1 < row2 => road_up_y < road_down_y
            if road_up_y <= agent_up_y <= road_down_y:
                # agent is inside the road
                closest_front_dist = 0.0
            elif road_down_y < agent_up_y and road_up_y < agent_up_y: 
                # agent is before the road
                closest_front_dist = min(closest_front_dist, agent_up_y - road_down_y)
            if road_up_y <= agent_down_y <= road_down_y:
                # agent is inside the road
                closest_back_dist = 0.0
            elif agent_down_y < road_up_y and agent_down_y < road_down_y:
                # agent is after the road
                closest_back_dist = min(closest_back_dist, road_up_y - agent_down_y)

        closest_front_dist = min(max(closest_front_dist, 0.0), 1.0)
        closest_back_dist = min(max(closest_back_dist, 0.0), 1.0)
        return closest_front_dist, closest_back_dist

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
