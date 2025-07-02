import math
import pygame

from pedestrian_env.envs.crosswalk import CrossWalks, RowType
from pedestrian_env.envs.game_object import Agent, Car, Cars

class World:
    ROAD_GRAY_COLOR = (89, 89, 89)
    ROAD_WHITE_COLOR = (250, 250, 250)
    SAFE_WHITE_COLOR = (217, 217, 217)

    def __init__(self, map_grid_width, map_grid_height, pix_square_size, steps_per_second, random, debug):
        self.random = random
        self.map_grid_width = map_grid_width
        self.map_grid_height = map_grid_height
        self.pix_square_size = pix_square_size
        self.map_width = self.map_grid_width * self.pix_square_size
        self.map_height = self.map_grid_height * self.pix_square_size

        self.steps_per_second = steps_per_second
        self.agent = Agent(map_grid_width, map_grid_height, pix_square_size, steps_per_second, debug)
        self.initial_player_y = self.agent.get_cur_y()

        self._generate_rows(map_grid_height)
        self.crosswalks = CrossWalks.generate_crosswalks(self.agent, self.row_types, self.random)
        self.cars = Cars.generate_cars(self.agent, self.row_types, self.max_height_dic, self.crosswalks,
                                       pix_square_size, map_grid_width, map_grid_height, steps_per_second, random)

    def target_lane_reached(self):
        cur_y = self.agent.get_cur_y()
        return cur_y == Agent.TARGET_LANE

    def update_positions(self, dt):
        self.agent.update_position(dt)
        self.crosswalks.update_activation(self.agent)
        self.cars.update_positions(dt)

    def calculate_up_rewards(self):
        return int(self.steps_per_second * (self.initial_player_y - self.agent.get_target_y()))

    def render(self):
        canvas = pygame.Surface((self.map_width, self.map_height))
        canvas.fill(self.ROAD_GRAY_COLOR)

        adjustment = 0.5 * self.pix_square_size
        for safe_row_idx in self.safe_row_idx_list:
            pygame.draw.rect(canvas, self.SAFE_WHITE_COLOR, (0, safe_row_idx * self.pix_square_size - adjustment, self.map_width, self.pix_square_size + 1))

        for other_direction_idx in self.other_direction_boundary_idx_list:
            start_x = 0
            pygame.draw.line(
                canvas,
                self.ROAD_WHITE_COLOR,
                (start_x, self.pix_square_size * other_direction_idx + adjustment),
                (self.map_width, self.pix_square_size * other_direction_idx + adjustment),
                width=3,
            )

        for boundary_idx in self.lane_boundary_idx_list:
            for x in range(0, self.map_grid_width, 5):
                start_x = x * self.pix_square_size
                pygame.draw.line(
                    canvas,
                    self.ROAD_WHITE_COLOR,
                    (start_x, self.pix_square_size * boundary_idx + adjustment),
                    (start_x + 2 * self.pix_square_size, self.pix_square_size * boundary_idx + adjustment),
                    width=6,
                )

        for crosswalk in self.crosswalks.elements:
            start_x = crosswalk.col * self.pix_square_size - adjustment
            start_y = crosswalk.row1 * self.pix_square_size - adjustment
            end_y = crosswalk.row2 * self.pix_square_size + adjustment
            # NOTE: cover up background (-1 pixel at top and bottom, +self.pix_square_size at left and right)
            pygame.draw.rect(canvas, self.ROAD_GRAY_COLOR, (start_x - self.pix_square_size, start_y + 1, self.pix_square_size * 3, end_y - start_y - 2))

            stripe_count = 3 * (crosswalk.row2 - crosswalk.row1 + 1)
            stripe_thickness = self.pix_square_size / 3
            for i in range(stripe_count):
                for j in range(2):
                    if i % 2 == j: continue
                    stripe_rect = pygame.Rect(start_x + (self.pix_square_size/2 * j), start_y + i * stripe_thickness, self.pix_square_size/2, stripe_thickness)
                    pygame.draw.rect(canvas, self.ROAD_WHITE_COLOR, stripe_rect)

        rendered_agent_position = self.agent.render(canvas)
        self.cars.render(canvas)

        return canvas, rendered_agent_position

    def _generate_rows(self, map_grid_height, max_consecutive_danger_lanes=4):
        rows = []
        consecutive_danger_lanes = 0
        while len(rows) < map_grid_height - 2:
            available_rows = (map_grid_height - 2) - len(rows)
            if available_rows >= Car.MAX_HEIGHT and consecutive_danger_lanes < max_consecutive_danger_lanes:
                row_type = self.random.choice([RowType.SAFE, 1, 2])
                if row_type == 1:
                    rows.append(RowType.CAR_GOING_RIGHT)
                    rows.append(RowType.CAR_GOING_RIGHT)
                    consecutive_danger_lanes += 2
                    continue
                elif row_type == 2:
                    rows.append(RowType.CAR_GOING_LEFT)
                    rows.append(RowType.CAR_GOING_LEFT)
                    consecutive_danger_lanes += 2
                    continue
            rows.append(RowType.SAFE)
            consecutive_danger_lanes = 0

        total_rows = [RowType.SAFE] + rows + [RowType.SAFE] # target area + lanes + starting area

        self.safe_row_idx_list = []
        for idx in range(len(total_rows)):
            if total_rows[idx] == RowType.SAFE:
                self.safe_row_idx_list.append(idx)

        self.lane_boundary_idx_list = []
        for idx in range(len(total_rows)-1):
            if total_rows[idx] != RowType.SAFE and total_rows[idx+1] != RowType.SAFE:
                self.lane_boundary_idx_list.append(idx)

        self.other_direction_boundary_idx_list = []
        for idx in range(len(total_rows)-1):
            if total_rows[idx] == RowType.CAR_GOING_RIGHT and total_rows[idx+1] == RowType.CAR_GOING_LEFT:
                self.other_direction_boundary_idx_list.append(idx)
                self.lane_boundary_idx_list.remove(idx)
            if total_rows[idx] == RowType.CAR_GOING_LEFT and total_rows[idx+1] == RowType.CAR_GOING_RIGHT:
                self.other_direction_boundary_idx_list.append(idx)
                self.lane_boundary_idx_list.remove(idx)

        self.max_height_dic = {}
        for row_idx in range(len(total_rows)):
            if total_rows[row_idx] == RowType.SAFE: continue
            self.max_height_dic[row_idx] = 1
            prev_row_idx = row_idx - 1
            while prev_row_idx >= 0:
                if total_rows[prev_row_idx] != total_rows[row_idx]: break
                self.max_height_dic[prev_row_idx] = min(self.max_height_dic[prev_row_idx] + 1, Car.MAX_HEIGHT)
                prev_row_idx -= 1
        self.row_types = total_rows
