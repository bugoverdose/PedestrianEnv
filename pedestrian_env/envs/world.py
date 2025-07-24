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

        self.grid_type_dict = [[GridType.DANGER for _ in range(map_grid_width)] for _ in range(map_grid_height)]
        for y in self.roads.safe_row_idx_list:
            for x in range(map_grid_width):
                self.grid_type_dict[y][x] = GridType.SAFE

        for road in self.roads.elements:
            crosswalk = road.crosswalk
            if crosswalk is None: continue
            for y in range(crosswalk.top_row, crosswalk.end_row+2):
                for x in range(crosswalk.left_end, crosswalk.right_end+1):
                    self.grid_type_dict[y][x] = GridType.CROSSWALK

        for y in range(map_grid_height):
            for x in range(map_grid_width):
                if agent_x_range[0] <= x <= agent_x_range[1]: continue
                self.grid_type_dict[y][x] = GridType.UNREACHABLE

    def target_lane_reached(self):
        cur_y = self.agent.cur_location[1]
        return cur_y == Agent.TARGET_LANE

    def nearby_road_info(self, game_screen_visible_range, crosswalk_visible_range, max_dist = 99):
        [agent_x, agent_y] = self.agent.cur_location
        agent_head_y, agent_tail_y = agent_y - Agent.RADIUS, agent_y + Agent.RADIUS
        cur_road_start_dist, prev_road_end_dist = max_dist, max_dist
        cur_road, cur_road_found = None, False
        for road in self.roads.elements:
            road_up_y, road_down_y = road.row1 - 0.5, road.row2 + 0.5 # row1 < row2 => road_up_y < road_down_y
            if not cur_road_found:
                if road_up_y <= agent_head_y <= road_down_y:
                    # agent is inside the road
                    cur_road = road
                    cur_road_start_dist = agent_head_y - road_down_y
                    cur_road_found = True
                elif road_down_y < agent_head_y and road_up_y < agent_head_y: 
                    # agent is before the road
                    dist = agent_head_y - road_down_y
                    if dist < cur_road_start_dist:
                        cur_road = road
                        cur_road_start_dist = dist

            if agent_tail_y < road_up_y and agent_tail_y < road_down_y:
                # agent is after the road
                dist = road_up_y - agent_tail_y
                if dist < prev_road_end_dist:
                    prev_road_end_dist = dist
            elif road_up_y <= agent_tail_y <= road_down_y:
                # agent is inside the road
                cur_road = road

        cur_road_end_dist = 0 # no road in front
        cur_crosswalk_discovered, cur_crosswalk_x_diff = 0, 0 # not found
        car_penalty = None
        nearby_cars = []
        if cur_road is not None:
            cur_road_end_dist, cur_crosswalk_x_diff, cur_crosswalk_discovered, nearby_cars = cur_road.observe(self.agent, self.cars, game_screen_visible_range, crosswalk_visible_range)
            if cur_road.uid in self.agent.road_penalty_dict:
                car_penalty = self.agent.road_penalty_dict[cur_road.uid]
        return [prev_road_end_dist, cur_road_start_dist, cur_road_end_dist,
                cur_crosswalk_discovered, cur_crosswalk_x_diff,
                car_penalty, nearby_cars]

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
