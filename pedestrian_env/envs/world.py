import pygame

from pedestrian_env.envs.road import Roads
from pedestrian_env.envs.game_object import Agent, Cars

class World:
    ROAD_GRAY_COLOR = (89, 89, 89)
    ROAD_WHITE_COLOR = (250, 250, 250)
    SAFE_WHITE_COLOR = (217, 217, 217)

    def __init__(self, map_grid_width, map_grid_height, camera_size, pix_square_size, steps_per_second, random, debug):
        self.random = random
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.pix_square_size = pix_square_size
        self.map_width = self.map_grid_width * self.pix_square_size
        self.map_height = self.map_grid_height * self.pix_square_size

        self.steps_per_second = steps_per_second
        self.agent = Agent(map_grid_width, map_grid_height, pix_square_size, steps_per_second, debug)
        self.initial_player_y = self.agent.get_cur_y()

        self.roads = Roads(self.agent, camera_size, map_grid_height, self.random)
        self.cars = Cars.generate_cars(self.agent, self.roads, pix_square_size, map_grid_width, steps_per_second, random)

    def target_lane_reached(self):
        cur_y = self.agent.get_cur_y()
        return cur_y == Agent.TARGET_LANE

    def update_positions(self, dt):
        self.agent.update_position(dt)
        self.roads.update_crosswalk_activation()
        self.cars.update_positions(dt)

    def calculate_up_rewards(self):
        return int(self.steps_per_second * (self.initial_player_y - self.agent.get_target_y()))

    def render(self):
        canvas = pygame.Surface((self.map_width, self.map_height))
        canvas.fill(self.ROAD_GRAY_COLOR)

        adjustment = 0.5 * self.pix_square_size
        for safe_row_idx in self.roads.safe_row_idx_list:
            pygame.draw.rect(canvas, self.SAFE_WHITE_COLOR, (0, safe_row_idx * self.pix_square_size - adjustment, self.map_width, self.pix_square_size + 1))

        for other_direction_idx in self.roads.other_direction_boundary_idx_list:
            start_x = 0
            pygame.draw.line(
                canvas,
                self.ROAD_WHITE_COLOR,
                (start_x, self.pix_square_size * other_direction_idx + adjustment),
                (self.map_width, self.pix_square_size * other_direction_idx + adjustment),
                width=3,
            )

        for boundary_idx in self.roads.lane_boundary_idx_list:
            for x in range(0, self.map_grid_width, 5):
                start_x = x * self.pix_square_size
                pygame.draw.line(
                    canvas,
                    self.ROAD_WHITE_COLOR,
                    (start_x, self.pix_square_size * boundary_idx + adjustment),
                    (start_x + 2 * self.pix_square_size, self.pix_square_size * boundary_idx + adjustment),
                    width=6,
                )

        # add crosswalks
        for road in self.roads.elements:
            crosswalk = road.crosswalk
            if crosswalk is None: continue
            start_x = crosswalk.col * self.pix_square_size - adjustment
            start_y = road.row1 * self.pix_square_size - adjustment
            end_y = road.row2 * self.pix_square_size + adjustment
            # NOTE: cover up background (-1 pixel at top and bottom, +self.pix_square_size at left and right)
            pygame.draw.rect(canvas, self.ROAD_GRAY_COLOR, (start_x - self.pix_square_size, start_y + 1, self.pix_square_size * 3, end_y - start_y - 2))

            stripe_count = 3 * (road.row2 - road.row1 + 1)
            stripe_thickness = self.pix_square_size / 3
            for i in range(stripe_count):
                for j in range(2):
                    if i % 2 == j: continue
                    stripe_rect = pygame.Rect(start_x + (self.pix_square_size/2 * j), start_y + i * stripe_thickness, self.pix_square_size/2, stripe_thickness)
                    pygame.draw.rect(canvas, self.ROAD_WHITE_COLOR, stripe_rect)

        rendered_agent_position = self.agent.render(canvas)
        self.cars.render(canvas)

        return canvas, rendered_agent_position
