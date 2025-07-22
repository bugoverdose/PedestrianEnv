import gymnasium as gym

import pygame
import numpy as np

from pedestrian_env.envs.world import World
from pedestrian_env.envs.road import Roads, CrossWalk
from pedestrian_env.envs.game_object import Car
from pedestrian_env.envs.car_details import get_max_car_grid_width, get_panalty_range

class PedestrianEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    OFF_SCREEN_BLACK_COLOR = (0, 0, 0)
    UI_TEXT_WHITE_COLOR = (255, 255, 255)
    AGENT_DEAD_TEXT_COLOR = (255, 0, 0)
    SUCCESS_TEXT_COLOR = (0, 255, 0)
    TIME_OVER_TEXT_COLOR = (0, 0, 255)

    EXTRA_WIDTH = 500
    EXTRA_HEIGHT = 200

    DEFAULT_GAMEOVER_REST_TIME = 5_000
    BONUS_SCORE_PER_SEC = 50
    TIME_OVER_ALERT_SEC = 10

    def __init__(self, title="Pedestrian Task", width=25, height=20, camera_size=7, render_mode=None, tick_on_render=False, steps_per_second = 10, episode_duration_sec=30, debug=False, render_sprite=False):
        if width < 12: raise Exception("minimum width is 13")
        if height < 5: raise Exception("minimum height is 5")
        if episode_duration_sec < 10: raise Exception("minimum episode_duration_sec is 10")
        self.title = title
        self.map_grid_width = width
        self.map_grid_height = height + 1 # add starting lane
        self.tick_on_render = tick_on_render
        self.steps_per_second = steps_per_second
        self.step_ms =  1000 / self.steps_per_second # default: step once every 100ms
        self.metadata["render_fps"] = 60 # NOTE: must render multiple times between each step
        game_window_size = 2048
        self.pix_square_size = max(80, (game_window_size / max(self.map_grid_width, self.map_grid_height))) # The size of a single grid square in pixels
        self.camera_size = camera_size
        self.camera_size_pixel = camera_size * self.pix_square_size
        self.map_width = self.map_grid_width * self.pix_square_size
        self.map_height = self.map_grid_height * self.pix_square_size
        self.debug = debug
        self.render_sprite = render_sprite
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # game info
        self.world = None
        self.prev_rewards = 0 # sum of all the rewards from all the previous episodes
        self.cur_rewards = 0 # reward from the current ongoing episode
        self.best_rewards = 0
        self.GAME_TIME_MS = (episode_duration_sec + 1) * 1000
        self.time_left = self.GAME_TIME_MS
        self.game_over = False
        self.game_end_extra_score = 0

        # action
        self.action_space = gym.spaces.Discrete(5)
        # obs
        self._define_observation_space()

    def reset(self, seed=None, options=None):
        # NOTE: following line is needed for self.np_random
        super().reset(seed=seed)

        self.best_rewards = max(self.best_rewards, self.cur_rewards)
        self.prev_rewards += self.cur_rewards
        self.cur_rewards = 0
        self.time_left = self.GAME_TIME_MS
        self.game_over = False
        self.game_end_extra_score = 0

        self.world = World((self.agent_min_x, self.agent_max_x), self.map_grid_width, self.map_grid_height, self.camera_size, self.pix_square_size, self.steps_per_second, self.np_random, self.debug, self.render_sprite)

        if self.render_mode == "human":
            self.render()

        if self.debug:
            print(self.world.cars)

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call `reset` to
        reset this environment's state for the next episode.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barto Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
        """
        prev_up_rewards = self.world.calculate_up_rewards()
        self.world.agent.update_target(action)

        cur_up_rewards = self.world.calculate_up_rewards()
        reward = cur_up_rewards - prev_up_rewards
        agent_dead, death_penalty = self.world.cars.has_hit_agent()
        if agent_dead:
            terminated = True
            self.world.agent.set_dead()
            reward -= death_penalty
            self.game_end_extra_score = -death_penalty
        elif self._get_time_left_sec() <= 0:
            terminated = True
        else:
            # An episode is finished if the agent has reached the target lane
            terminated = self.world.target_lane_reached()
            if terminated:
                game_end_extra_score = self._get_time_left_sec()
                reward += game_end_extra_score * self.BONUS_SCORE_PER_SEC
                self.game_end_extra_score = game_end_extra_score * self.BONUS_SCORE_PER_SEC
        self.game_over = terminated
        self.cur_rewards += reward

        if self.render_mode != "human":
            self.update_positions(self.step_ms)
            self.time_left -= self.step_ms

        return self._get_obs(), reward, terminated, False, self._get_info()

    def clock_tick(self):
        return self.clock.tick(self.metadata["render_fps"])

    def update_positions(self, dt):
        self.world.update_positions(dt)

    def update_time_left(self, elapsed_time):
        if self.game_over: return
        self.time_left = self.GAME_TIME_MS - elapsed_time

    def render(self):
        if self.render_mode is None: return
        total_window_width = self.camera_size_pixel + self.EXTRA_WIDTH
        total_window_height = self.camera_size_pixel + self.EXTRA_HEIGHT

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption(self.title)
                pygame.display.init()
                self.window = pygame.display.set_mode((total_window_width, total_window_height))
            elif self.render_mode == "rgb_array":
                self.window = pygame.Surface((total_window_width, total_window_height))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas, rendered_agent_position = self.world.render()

        # clear background area
        bg_rect = pygame.Rect(0, 0, total_window_width, total_window_height)
        pygame.draw.rect(self.window, self.OFF_SCREEN_BLACK_COLOR, bg_rect)

        camera_rect = pygame.Rect(0, 0, self.camera_size_pixel, self.camera_size_pixel)
        camera_rect.center = rendered_agent_position
        # camera_rect.clamp_ip(canvas.get_rect()) # prevent camera from going out-of-bounds

        font_size_s = 32
        font_size_m = 48
        font_size_l = 80
        font_size_xl = 128
        center_x = total_window_width/2
        x_padding = 60
        left_ui_x = (self.EXTRA_WIDTH/2) - x_padding
        center_ui_x = center_x - x_padding
        right_ui_x = total_window_width - (self.EXTRA_WIDTH/2) - x_padding

        top_y = self.EXTRA_HEIGHT/2
        bottom_y = total_window_height - (self.EXTRA_HEIGHT/2) + 5
        left_game_x = self.EXTRA_WIDTH/2
        right_game_x = total_window_width - (self.EXTRA_WIDTH/2)
        top_game_y = self.EXTRA_HEIGHT/2

        # add game screen
        self.window.blit(canvas, (left_game_x, top_game_y), area=camera_rect)
        
        score_text_color = self.UI_TEXT_WHITE_COLOR
        if self.game_over:
            # add gameover screen
            dark_screen = pygame.Surface((self.map_width, self.map_height), pygame.SRCALPHA)
            alpha = 180 # 128 == 50% transparency
            dark_screen.fill((0, 0, 0, alpha))
            self.window.blit(dark_screen, (left_game_x, top_game_y), area=camera_rect)

            if self.game_end_extra_score < 0:
                game_status_desc = "YOU DIED"
                game_status_color = self.AGENT_DEAD_TEXT_COLOR
                score_text_color = self.AGENT_DEAD_TEXT_COLOR
            elif self.game_end_extra_score > 0:
                game_status_desc = "BONUS"
                game_status_color = self.SUCCESS_TEXT_COLOR
                score_text_color = self.SUCCESS_TEXT_COLOR
            else:
                game_status_desc = "TIME OVER"
                game_status_color = self.TIME_OVER_TEXT_COLOR

            test_render_start_height = total_window_height/2
            self._render_text(center_x, test_render_start_height, game_status_desc, game_status_color, font_size_xl, order="center")
            test_render_start_height += font_size_xl/2

            if self.game_end_extra_score != 0:
                text = f"+{self.game_end_extra_score}" if self.game_end_extra_score > 0 else f"{self.game_end_extra_score}"
                self._render_text(center_x, test_render_start_height, text, game_status_color, font_size_l, order="center")
                test_render_start_height += font_size_l/2

            # show score results
            center_left_ui_x = left_game_x + 80
            center_right_ui_x = right_game_x - 130

            # update current score (include game_end_extra_score)
            self._render_text(center_left_ui_x, test_render_start_height, f"Current Score:", score_text_color, font_size_m, order="left")
            self._render_text(center_right_ui_x, test_render_start_height, f"{self.cur_rewards}", score_text_color, font_size_m, order="right")
            test_render_start_height += font_size_m

            # update best score
            best_renewed = self.cur_rewards > self.best_rewards
            best_score_text_color = self.UI_TEXT_WHITE_COLOR
            if best_renewed and self.game_end_extra_score > 0:
                best_score_text_color = self.SUCCESS_TEXT_COLOR
            best_score = self.cur_rewards if best_renewed else self.best_rewards
            self._render_text(center_left_ui_x, test_render_start_height, f"Best Score:", best_score_text_color, font_size_m, order="left")
            self._render_text(center_right_ui_x, test_render_start_height, f"{best_score}", best_score_text_color, font_size_m, order="right")
            if best_renewed:
                self._render_text(center_right_ui_x + 10, test_render_start_height, "(NEW!)", best_score_text_color, font_size_m)
            test_render_start_height += font_size_m

            # update total score
            self._render_text(center_left_ui_x, test_render_start_height, f"Total Score:", self.UI_TEXT_WHITE_COLOR, font_size_m, order="left")
            self._render_text(center_right_ui_x, test_render_start_height, f"{self.prev_rewards + self.cur_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_m, order="right")

            # update time left
            time_left_sec = self._get_time_left_sec()
            time_left_sec_color = self.UI_TEXT_WHITE_COLOR if time_left_sec > self.TIME_OVER_ALERT_SEC else self.AGENT_DEAD_TEXT_COLOR
            self._render_text(right_game_x - 30, top_y + 30, f"Time Left: {time_left_sec}", time_left_sec_color, font_size_m, order="right")
        else:
            # add ingame UI

            # update current score (include game_end_extra_score)
            self._render_text(center_ui_x, bottom_y, f"Current Score: {self.cur_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_s)

            # update best score
            self._render_text(right_ui_x, bottom_y, f"Best Score: {self.best_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_s)

            # update total score: show the sum of previous scores (because it's confusing when both cur rewards and total rewards are constantly changing)
            self._render_text(left_ui_x, bottom_y, f"Total Score: {self.prev_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_s)

            # update time left
            time_left_sec = self._get_time_left_sec()
            time_left_sec_color = self.UI_TEXT_WHITE_COLOR if time_left_sec > self.TIME_OVER_ALERT_SEC else self.AGENT_DEAD_TEXT_COLOR
            self._render_text(right_ui_x, top_y - font_size_s - 5, f"Time Left: {time_left_sec}", time_left_sec_color, font_size_s)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            # needed to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            if self.tick_on_render:
                self.clock_tick()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def render_game_over(self):
        elapsed = 0
        total_elapsed = 0
        while total_elapsed < self.DEFAULT_GAMEOVER_REST_TIME:
            dt = self.clock_tick()
            elapsed += dt
            total_elapsed += dt
            while elapsed >= self.step_ms:
                elapsed -= self.step_ms
                agent_dead, _ = self.world.cars.has_hit_agent()
                if agent_dead:
                    self.world.agent.set_dead()
            self.update_positions(dt)
            self.render()

    def _define_observation_space(self):
        agent_x_buffer = 2 + int(max(get_max_car_grid_width(), self.camera_size)/2)
        self.agent_min_x = (1 + agent_x_buffer)
        self.agent_max_x = self.map_grid_width - 1 - agent_x_buffer
        agent_min_Y = 0
        agent_max_Y = self.map_grid_height - 1
        self.game_screen_visible_range = self.camera_size/2
        self.crosswalk_visible_range = self.game_screen_visible_range + CrossWalk.VISIBLE_WIDTH
        [min_car_penalty, max_car_penalty] = get_panalty_range()
        self.min_car_penalty = min_car_penalty
        self.max_car_penalty = max_car_penalty
        max_car_speed = max(Car.CAR_SPEEDS)
        self.observation_space = gym.spaces.Box(
            low=np.array([
                0,
                self.agent_min_x, agent_min_Y,
                -Roads.MAX_ROAD_SIZE, -Roads.MAX_ROAD_SIZE, 0.0, 
                0, -self.crosswalk_visible_range,
                0, self.min_car_penalty,
                -self.game_screen_visible_range, -max_car_speed,
                -self.game_screen_visible_range, -max_car_speed,
                -self.game_screen_visible_range, -max_car_speed,
                -self.game_screen_visible_range, -max_car_speed,
            ]),
            high=np.array([
                self.GAME_TIME_MS - 1000,
                self.agent_max_x, agent_max_Y, 
                0.0, Roads.MAX_ROAD_SIZE, Roads.MAX_ROAD_SIZE + 1,
                1, self.crosswalk_visible_range,
                1, self.max_car_penalty,
                self.game_screen_visible_range, max_car_speed,
                self.game_screen_visible_range, max_car_speed,
                self.game_screen_visible_range, max_car_speed,
                self.game_screen_visible_range, max_car_speed,
            ]),
            dtype=np.float32
        )

    def _get_obs(self):
        """
        Observation
        
        agent position
        [0] time left in ms (0 ~ 30000)
        [1] agent x (6.0 ~ 19.0)
        [2] agent y (0.0 ~ 20.0)

        road
        [3] y distance until tail entering previously crossed road (-4.0 ~ 0.0)
        [4] y distance until head entering start of the current/next road (-4.0 ~ 0.0: body inside the road, 0.0 ~ 4.0: before entering the road)
        [5] y distance until tail escaping end of the current/next road (0.0 ~ 5.0)

        crosswalk (NOTE: crosswalk visible range(4.5) == camera range (3.5) + visible width(1))
        [6] visible crosswalk in current/next road (0 = False, 1 = True)
        [7] x distance until discovered crosswalk of current/next road (-4.5 ~ 4.5)

        cars per lane inside current/next road
        [8] car penalty visible (0 = False, 1 = True)
        [9] car penalty (max value if unknown) (100, 500, 1000)
        [10] lane1: closest car x distance (-3.5 ~ 0: going right, 0 ~ 3.5: going left)
        [11] lane1: closest car speed (-4.5 ~ -3.0: going left, 0: stopped, 3.0 ~ 4.5: going right)
        [12] lane2: closest car x distance (-3.5 ~ 0: going right, 0 ~ 3.5: going left)
        [13] lane2: closest car speed (-4.5 ~ -3.0: going left, 0: stopped, 3.0 ~ 4.5: going right)
        [14] lane3: closest car x distance (-3.5 ~ 0: going right, 0 ~ 3.5: going left)
        [15] lane3: closest car speed (-4.5 ~ -3.0: going left, 0: stopped, 3.0 ~ 4.5: going right)
        [16] lane4: closest car x distance (-3.5 ~ 0: going right, 0 ~ 3.5: going left)
        [17] lane4: closest car speed (-4.5 ~ -3.0: going left, 0: stopped, 3.0 ~ 4.5: going right)
        """
        time_left = max(0, self.time_left - 1000)
        agent_x, agent_y = self.world.agent.get_cur_location_rounded()

        [prev_road_end_dist, cur_road_start_dist, cur_road_end_dist,
         cur_crosswalk_discovered, cur_crosswalk_x_diff,
         car_penalty, nearby_cars] = self.world.nearby_road_info(self.game_screen_visible_range, self.crosswalk_visible_range)

        prev_road_end_dist = min(max(prev_road_end_dist, -Roads.MAX_ROAD_SIZE), 0)
        cur_road_start_dist = min(max(cur_road_start_dist, -Roads.MAX_ROAD_SIZE), Roads.MAX_ROAD_SIZE)
        cur_road_end_dist = min(max(cur_road_end_dist, 0), Roads.MAX_ROAD_SIZE + 1)

        car_penalty_is_visible = 1.0 if car_penalty is not None else 0.0
        visible_car_penalty = car_penalty if car_penalty is not None else self.max_car_penalty
        if len(nearby_cars) < 8:
            nearby_cars += [-self.game_screen_visible_range, 0] * int((8 - len(nearby_cars)) / 2)
            if len(nearby_cars) != 8:
                raise Exception("invalid implementation")
        return np.array([time_left,
                         agent_x, agent_y,
                         prev_road_end_dist, cur_road_start_dist, cur_road_end_dist,
                         cur_crosswalk_discovered, cur_crosswalk_x_diff,
                         car_penalty_is_visible, visible_car_penalty,
                         nearby_cars[0], nearby_cars[1], nearby_cars[2], nearby_cars[3],
                         nearby_cars[4], nearby_cars[5], nearby_cars[6], nearby_cars[7]
        ])

    def _get_info(self):
        """
        Returns:
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
        """
        agent_x, agent_y = self.world.agent.get_cur_location_rounded()
        return {
            "agent_x": agent_x,
            "agent_y": agent_y,
            "is_dead": self.world.agent.is_dead,
            "game_end_extra_score": self.game_end_extra_score,
            "cur_episode_score": self.cur_rewards,
            "total_score": self._total_rewards(),
        }

    def _total_rewards(self):
        return self.prev_rewards + self.cur_rewards

    def _get_time_left_sec(self):
        return int(self.time_left/1000)

    def _render_text(self, x, y, text, color, font_size, order=None):
        font = pygame.font.SysFont(None, font_size)
        text_surface = font.render(text, True, color)
        if order == "center":
            text_location = text_surface.get_rect(center=(x, y))
        elif order == "left":
            text_location = text_surface.get_rect(topleft=(x, y))   
        elif order == "right":
            text_location = text_surface.get_rect(topright=(x, y))            
        else:
            text_location = (x, y)
        self.window.blit(text_surface, text_location)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

