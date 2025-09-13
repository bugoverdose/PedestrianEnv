import gymnasium as gym

import pygame
import numpy as np

from pedestrian_env.envs.world import World
from pedestrian_env.envs.game_object import Car, load_player_asset_info
from pedestrian_env.envs.car_details import Penalty, get_max_car_grid_width, load_car_details_dict
from pedestrian_env.envs.action import Action, ACTION_TO_DELTA, ACTION_DURATION
from pedestrian_env.envs.utils import is_overlapping, is_overlapping_rectangles

class PedestrianEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    OFF_SCREEN_BLACK_COLOR = (0, 0, 0)
    UI_TEXT_WHITE_COLOR = (255, 255, 255)
    AGENT_DEAD_TEXT_COLOR = (255, 0, 0)
    SUCCESS_TEXT_COLOR = (0, 255, 0)
    TIME_OVER_TEXT_COLOR = (0, 0, 255)

    EXTRA_WIDTH = 500
    EXTRA_HEIGHT = 200

    BONUS_SCORE_PER_SEC = 50
    TIME_OVER_ALERT_SEC = 10

    def __init__(self,
                 title="Pedestrian Task",
                 width=25,
                 height=21,
                 camera_width=11,
                 camera_height=7,
                 steps_per_second=10,
                 gamescreen_width_fixed=False,
                 episode_duration_sec=30,
                 gameover_screen_time=5000,
                 fixed_episode_seed_range=None,
                 render_mode=None,
                 realtime=False,
                 extra_reward_using_crosswalk=False,
                 debug=False):
        if width < 12: raise Exception("minimum width is 13")
        if height < 6: raise Exception("minimum height is 6")
        if camera_height < 7: raise Exception("minimum camera_height is 7")
        if camera_height%2 == 0: raise Exception("minimum camera_height should be an odd number")
        if episode_duration_sec < 10: raise Exception("minimum episode_duration_sec is 10")
        self.title = title
        self.map_grid_width = width
        self.map_grid_height = height
        self.steps_per_second = steps_per_second
        self.step_ms =  1000 / self.steps_per_second # default: step once every 100ms
        self.realtime = realtime
        self.gameover_screen_time = gameover_screen_time
        self.metadata["render_fps"] = 60 # NOTE: must render multiple times between each step
        game_window_size = 2048
        self.pix_square_size = max(80, (game_window_size / max(self.map_grid_width, self.map_grid_height))) # The size of a single grid square in pixels
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_width_pixel = camera_width * self.pix_square_size
        self.camera_height_pixel = camera_height * self.pix_square_size
        self.map_width = self.map_grid_width * self.pix_square_size
        self.map_height = self.map_grid_height * self.pix_square_size
        self.extra_reward_using_crosswalk = extra_reward_using_crosswalk
        self.gamescreen_width_fixed = gamescreen_width_fixed
        assert fixed_episode_seed_range is None or len(fixed_episode_seed_range) > 0
        self.fixed_episode_seed_range = fixed_episode_seed_range
        self.fixed_episode_seed_idx = 0
        self.debug = debug
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_sprite = render_mode is not None
        self.player_asset_info = load_player_asset_info(self.pix_square_size, self.render_sprite)
        self.car_details_dict = load_car_details_dict(self.pix_square_size, self.render_sprite)
        self.window = None
        self.clock = None

        # game info
        self.world = None
        self.cur_seed = None
        self.prev_rewards = 0 # sum of all the rewards from all the previous episodes
        self.cur_rewards = 0 # reward from the current ongoing episode
        self.best_rewards = 0
        self.GAME_TIME_MS = (episode_duration_sec + 1) * 1000
        self.time_left = self.GAME_TIME_MS
        self.elapsed = 0
        self.game_over = False
        self.game_end_extra_score = 0
        self.real_time_step_passed = 0
        self.obs = None
        self.info = None

        self.max_car_penalty = Penalty.HIGH.value
        self.max_car_speed = max(Car.CAR_SPEEDS)
        agent_x_buffer = 2 + int(max(get_max_car_grid_width(), self.camera_width)/2)
        agent_min_x = (agent_x_buffer)
        agent_max_x = self.map_grid_width - 1 - agent_x_buffer
        self.agent_x_range = (agent_min_x, agent_max_x)

        # action
        self.action_space = gym.spaces.Discrete(5)
        # obs
        self._define_observation_space()

    def reset(self, seed=None):
        # set seed for World initialization and car speed change
        self.cur_seed = seed
        if seed is None and self.fixed_episode_seed_range is not None:
            # use each seed in the fixed_episode_seed_range
            self.cur_seed = self.fixed_episode_seed_range[self.fixed_episode_seed_idx]
            self.fixed_episode_seed_idx = (self.fixed_episode_seed_idx + 1) % len(self.fixed_episode_seed_range)
        super().reset(seed=self.cur_seed)
        self.elapsed = 0

        self.best_rewards = max(self.best_rewards, self.cur_rewards)
        self.prev_rewards += self.cur_rewards
        self.cur_rewards = 0
        self.time_left = self.GAME_TIME_MS
        self.game_over = False
        self.game_end_extra_score = 0
        self.real_time_step_passed = 0

        self.world = World(self.agent_x_range,
                           self.map_grid_width,
                           self.map_grid_height,
                           self.camera_width,
                           self.pix_square_size,
                           self.steps_per_second,
                           self.np_random,
                           self.debug,
                           self.player_asset_info,
                           self.car_details_dict)

        if self.render_sprite:
            self.render()

        if self.debug:
            print(self.world.cars)

        self.obs = None
        self.info = None
        self._update_step_state()
        return self.obs, self.info

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
        if self.extra_reward_using_crosswalk:
            prev_agent_x, prev_agent_y = self.world.agent.get_cur_location_grid()
        cumulative_reward = -self.world.calculate_cum_crossing_rewards()
        action_duration = ACTION_DURATION[action]
        self.real_time_step_passed += action_duration
        if not self.realtime:
            # interacting with the environment using RL algorithm
            self.world.agent.update_target(action)
            for _ in range(action_duration):
                reward, terminated = self._check_gameover()
                cumulative_reward += reward
                self.apply_time_and_render(self.step_ms)
                self.game_over = terminated
                if self.game_over: break
        else:
            # behaviour task for humans
            cur_count = 0
            self.world.agent.update_target(action)
            while True:
                if self.elapsed >= self.step_ms:
                    self.elapsed -= self.step_ms
                    if cur_count >= action_duration: break
                    cur_count += 1
                    reward, terminated = self._check_gameover()
                    cumulative_reward += reward
                    self.game_over = terminated
                    if self.game_over: break
                self.world.agent.mini_step_count = 0 if action == Action.NOTHING.value else cur_count
                dt = self.clock_tick()
                self.elapsed += dt
                self.apply_time_and_render(dt)

        # reward shaping to motivate usage of crosswalk
        if self.extra_reward_using_crosswalk:
            is_crosswalk = self.world.crosswalk_map
            agent_x, agent_y = self.world.agent.get_cur_location_grid()
            if action == Action.UP.value:
                if is_crosswalk[prev_agent_y][prev_agent_x]:
                    cumulative_reward += 10 # used the crosswalk
            elif action == Action.DOWN.value:
                if not is_crosswalk[prev_agent_y][prev_agent_x] and is_crosswalk[agent_y][agent_x]:
                    cumulative_reward -= 10 # went back to the crosswalk
            elif action in [Action.RIGHT.value, Action.LEFT.value]:
                if not is_crosswalk[prev_agent_y][prev_agent_x] and is_crosswalk[agent_y][agent_x]:
                    cumulative_reward += 20 # entered crosswalk
                if is_crosswalk[prev_agent_y][prev_agent_x] and not is_crosswalk[agent_y][agent_x]:
                    cumulative_reward -= 20 # leaved crosswalk

        # An episode is finished if the agent has reached the target lane
        if not self.world.agent.is_dead and self.world.target_lane_reached():
            self.game_over = True
            game_end_extra_score = self._get_time_left_sec()
            cumulative_reward += game_end_extra_score * self.BONUS_SCORE_PER_SEC
            self.game_end_extra_score = game_end_extra_score * self.BONUS_SCORE_PER_SEC

        cumulative_reward += self.world.calculate_cum_crossing_rewards()
        self.cur_rewards += cumulative_reward

        if self.game_over and self.realtime:
            self._render_game_over()
        
        self._update_step_state()
        return self.obs, cumulative_reward, self.game_over, False, self.info

    def _check_gameover(self):
        reward, terminated = 0, False
        agent_dead, death_penalty = self.world.check_and_update_agent_collision()
        if agent_dead:
            terminated = True
            reward -= death_penalty
            self.game_end_extra_score = -death_penalty
        elif self._get_time_left_sec() <= 0:
            self.world.agent.stop()
            terminated = True
        return reward, terminated

    def clock_tick(self):
        return self.clock.tick(self.metadata["render_fps"])

    def apply_time_and_render(self, dt):
        # NOTE: world should always keep moving continuously
        self.world.update_positions(dt)
        if not self.game_over:
            self.time_left -= dt
        self.render()

    def render(self):
        if self.render_mode is None: return
        total_window_width = self.camera_width_pixel + self.EXTRA_WIDTH
        total_window_height = self.camera_height_pixel + self.EXTRA_HEIGHT

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

        camera_rect = pygame.Rect(0, 0, self.camera_width_pixel, self.camera_height_pixel)
        if self.gamescreen_width_fixed:
            camera_rect.center = (self.world.agent.init_pos[0] * self.pix_square_size, rendered_agent_position[1] - (2 * self.pix_square_size))
        else:
            camera_rect.center = (rendered_agent_position[0], rendered_agent_position[1] - (2 * self.pix_square_size))
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
        dark_screen_alpha = 180 # 128 == 50% transparency

        # add game screen
        self.window.blit(canvas, (left_game_x, top_game_y), area=camera_rect)
        
        score_text_color = self.UI_TEXT_WHITE_COLOR
        if self.game_over:
            # add gameover screen
            dark_screen = pygame.Surface((self.map_width, self.map_height), pygame.SRCALPHA)
            dark_screen.fill((0, 0, 0, dark_screen_alpha))
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
            # unreachable area
            if not self.gamescreen_width_fixed:
                (agent_min_x, agent_max_x) = self.agent_x_range
                [agent_cur_x, _] = self.world.agent.cur_location
                game_screen_left_end = agent_cur_x - (self.camera_width // 2)
                game_screen_right_end = agent_cur_x + (self.camera_width // 2)
                buffer = 10
                if game_screen_left_end < agent_min_x:
                    unreachable_area_width = (agent_min_x - game_screen_left_end) * self.pix_square_size + buffer
                    dark_screen = pygame.Surface((unreachable_area_width, self.map_height), pygame.SRCALPHA)
                    dark_screen.fill((0, 0, 0, dark_screen_alpha))
                    self.window.blit(dark_screen, (left_game_x - buffer, top_game_y))
                elif agent_max_x < game_screen_right_end:
                    unreachable_area_width = (game_screen_right_end - agent_max_x) * self.pix_square_size + buffer
                    dark_screen = pygame.Surface((unreachable_area_width, self.map_height), pygame.SRCALPHA)
                    dark_screen.fill((0, 0, 0, dark_screen_alpha))
                    self.window.blit(dark_screen, (right_game_x - unreachable_area_width + buffer, top_game_y))

            # add ingame UI
            # update current score (include game_end_extra_score)
            self._render_text(center_ui_x, bottom_y, f"Current Score: {self.cur_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_s)

            # update total score: show the sum of previous scores (because it's confusing when both cur rewards and total rewards are constantly changing)
            self._render_text(right_ui_x, bottom_y, f"Total Score: {self.prev_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_s)

            # update best score
            self._render_text(left_ui_x, bottom_y, f"Best Score: {self.best_rewards}", self.UI_TEXT_WHITE_COLOR, font_size_s)

            # update time left
            time_left_sec = self._get_time_left_sec()
            time_left_sec_color = self.UI_TEXT_WHITE_COLOR if time_left_sec > self.TIME_OVER_ALERT_SEC else self.AGENT_DEAD_TEXT_COLOR
            self._render_text(right_ui_x, top_y - font_size_s - 5, f"Time Left: {time_left_sec}", time_left_sec_color, font_size_s)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def _render_game_over(self):
        elapsed = 0
        total_elapsed = 0
        while total_elapsed < self.gameover_screen_time:
            dt = self.clock_tick()
            elapsed += dt
            total_elapsed += dt
            while elapsed >= self.step_ms:
                elapsed -= self.step_ms
                # NOTE: handle death during time over
                self.world.check_and_update_agent_collision()
            self.apply_time_and_render(dt)

    def _define_observation_space(self):
        """
        # Observation space
        All the observations are within 0 ~ 1 or -1 ~ 1 range,
        and the total length varies depending on the camera_height configuration.

        ## Meta info [0 ~ 4]
        - play time left
          - 1 = initialized
          - 0 ~ 1 = time remaining
          - 0 = game over (time over, death, early finish)
        - closeness to unreachable area
          - 0 = only reachable area in sight
          - 0 ~ 1 = closeness to nearest unreachable tile in the direction
          - 1 = adjacent to unreachable tile, can't move in the direction
          - size: 4 (directions: up, down, right, left)

        ## Tile types near the agent [5 ~ 12]
        - is dangerous
          - 0 = safe zone or end of the map
          - 1 = on the road
          - size: 3 (current, up, down)
        - is crosswalk
          - 0 = not crosswalk
          - 1 = crosswalk or crosswalk activation tile in the safe zone
          - size: 5 (current, up, down, right, left)

        ## Target road info [13 ~ 19]
        Information about the target road that the agent is crossing or should cross
        - crosswalk visible
          - 0 = no crosswalk visible
          - 1 = crosswalks in sight
        - car penalty
          - 0.1, 0.5, 1.0 = penalty based on car color (100, 500, 1000)
        - y distance from the end of the road
          - 0.0 = can escape the road after moving up 1 more tile
          - 0.25 = can escape the road after moving up 2 more tile
          - 0.50 = can escape the road after moving up 3 more tile
          - 0.75 = can escape the road after moving up 4 more tile
          - 1.0 = in the safe zone, before crossing the road
        - y distance from the start of the road
          - 0.0 = in the safe zone, before crossing the road
          - 0.25 = agent can escape the road after moving down 1 more tile
          - 0.50 = agent can escape the road after moving down 2 more tile
          - 0.75 = agent can escape the road after moving down 3 more tile
          - 1.0 = agent can escape the road after moving down 4 more tile
        - total number of cars visible on the road
          - 0 ~ 1 = 0 ~ 6 cars in sight
          - NOTE: each road can have maximum of 6 cars
        - number of cars coming toward the agent
          - 0 ~ 1 = 0 ~ 6 cars in the direction
          - size: 4 (for each direction: up, down)

        ## Distance from crosswalks [20 ~ 28]
        - closeness to three nearest crosswalk
          - 0 = crosswalk not visible
          - 0 ~ 1 = closeness (1 - distance from the target crosswalk)
          - 1 = in the same column or row as the target crosswalk
          - size: 3 (for each direction: up, right, left) * 3 crosswalks

        ## Cars [29 ~ 64]
        Information about the nearest cars in each lane for each direction (left and right sides).
        Only consider the roads right next to the agent (1 in front of the agent, 1 behind the agent, or the 1 the agent is crossing)
        Each lane can only have 1 or 2 cars.
        Ignore the ones moving away since they are the same as out of sight.
        - closeness of nearest dangerous cars
          - 0 = no car coming toward the agent in the direction
          - 0 ~ 1 = closeness of the nearest car
          - 1 = in the same column as agent
          - size: 2 cars * 6 lanes * 2 (left vs right)
        - approaching speed of nearest dangerous cars
          - 0 = no car or stopped
          - 0 ~ 1 = speed of 3.0 ~ 4.5 coming toward the agent
          - size: 1 closest car * 6 lanes * 2 (left vs right)
        """
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0 for _ in range(65)]),
            high=np.array([1.0 for _ in range(65)]),
            dtype=np.float64
        )

    def _update_step_state(self):
        self._update_obs()
        self._update_info()

    def _update_obs(self, meta_info = True, tile_types = True, target_road_info = True, crosswalk_info = True, cars_info = True):
        obs = []
        agent_x, agent_y = self.world.agent.get_cur_location_grid()
        [_, _, agent_left_x, agent_right_x] = self.world.agent.get_hitbox()
        agent_y_range = [0, self.map_grid_height-1]
        y_top_visible_distance = self.camera_height//2 + 2
        y_bottom_visible_distance = self.camera_height//2 - 2
        grid_top_y = agent_y - y_top_visible_distance
        grid_bottom_y = grid_top_y + self.camera_height - 1
        x_visible_distance = self.camera_width//2
        if self.gamescreen_width_fixed:
            grid_x_left_end = self.world.agent.init_pos[0] - x_visible_distance
        else:
            grid_x_left_end = agent_x - x_visible_distance
        grid_x_right_end = grid_x_left_end + self.camera_width - 1

        visible_top_y_end = grid_top_y - 0.5
        visible_bottom_y_end = visible_top_y_end + self.camera_height
        visible_x_left_end = grid_x_left_end - 0.5
        visible_x_right_end = visible_x_left_end + self.camera_width
        game_screen_rect = [visible_top_y_end, visible_bottom_y_end, visible_x_left_end, visible_x_right_end]

        left_visible_dist = agent_left_x - visible_x_left_end
        right_visible_dist = visible_x_right_end - agent_right_x

        # Meta info
        if meta_info:
            time_left = 0
            if not self.game_over:
                time_left = max(0, (self.time_left - 1000) / (self.GAME_TIME_MS  - 1000))
            top_end_closeness = 0
            if grid_top_y < agent_y_range[0]:
                top_end_closeness = (y_top_visible_distance - (agent_y - agent_y_range[0])) / y_top_visible_distance
            bottom_end_closeness = 0
            if grid_bottom_y > agent_y_range[1]:
                bottom_end_closeness = (y_bottom_visible_distance - (agent_y_range[1] - agent_y)) / y_bottom_visible_distance
            left_end_closeness = 0
            if grid_x_left_end < self.agent_x_range[0]:
                left_end_closeness = (x_visible_distance - (agent_x - self.agent_x_range[0])) / x_visible_distance
            right_end_closeness = 0
            if self.agent_x_range[1] < grid_x_right_end:
                right_end_closeness = (x_visible_distance - (self.agent_x_range[1] - agent_x)) / x_visible_distance
            obs += [time_left, top_end_closeness, bottom_end_closeness, left_end_closeness, right_end_closeness]
            # print("agent:", agent_x, agent_y, self.camera_height, self.camera_width)
            # print("grid_top_y:", grid_top_y, " top_end_of_map_visible:", top_end_closeness, " distance:", agent_y - agent_y_range[0])
            # print("grid_bottom_y:", grid_bottom_y, " bottom_end_closeness:", bottom_end_closeness, " distance:", agent_y_range[1] - agent_y)
            # print("grid_x_left_end:", grid_x_left_end, ", left_end_closeness:", left_end_closeness, " distance:", agent_x - self.agent_x_range[0])
            # print("grid_x_right_end:", grid_x_right_end, ", right_end_closeness:", right_end_closeness, " distance:", self.agent_x_range[1] - agent_x)
        assert len(obs) == 5

        # Tile types near the agent
        if tile_types:
            is_dangerous = []
            is_crosswalk = []
            for action in [Action.NOTHING, Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]:
                [dx, dy] = ACTION_TO_DELTA[action.value]
                grid_x = agent_x + dx
                grid_y = agent_y + dy
                out_of_move_range = grid_y < agent_y_range[0] or grid_y > agent_y_range[1] or grid_x < self.agent_x_range[0] or grid_x > self.agent_x_range[1]
                if action != Action.RIGHT and action != Action.LEFT:
                    if out_of_move_range:
                        is_dangerous.append(0)
                    else:
                        is_dangerous.append(1 if self.world.danger_row[grid_y] else 0)
                if out_of_move_range:
                    is_crosswalk.append(0)
                else:
                    is_crosswalk.append(1 if self.world.crosswalk_map[grid_y][grid_x] else 0)
            obs += is_dangerous
            obs += is_crosswalk
            # print("is_dangerous cur:", is_dangerous[0])
            # print("is_dangerous up:", is_dangerous[1])
            # print("is_dangerous down:", is_dangerous[2])
            # print("is_crosswalk cur:", is_crosswalk[0])
            # print("is_crosswalk: up", is_crosswalk[1])
            # print("is_crosswalk down:", is_crosswalk[2])
            # print("is_crosswalk right:", is_crosswalk[3])
            # print("is_crosswalk left:", is_crosswalk[4])
        assert len(obs) == 13

        # Target road info
        prev_road = None # the road that the agent previously crossed
        target_road = None # the road that the agent is crossing or should cross
        for i in range(len(self.world.roads.elements), 0, -1):
            road = self.world.roads.elements[i-1]
            if road.top_y > agent_y:
                prev_road = road
                continue # already crossed
            target_road = road
            break
        if target_road_info:
            crosswalk_visible = 0.0
            penalty = 0.0
            dist_end_of_target_road = 1.0
            dist_start_of_target_road = 0.0
            visible_car_count = 0
            dangerous_car_counts = [0, 0]
            if target_road is not None:
                if target_road.crosswalk is not None:
                    if is_overlapping(target_road.crosswalk.left_end, target_road.crosswalk.right_end, grid_x_left_end, grid_x_right_end):
                        crosswalk_visible = 1.0
                penalty = road.risk_detail.penalty.value / self.max_car_penalty
                is_crossing = road.bottom_y >= agent_y >= road.top_y
                if is_crossing:
                    dist_end_of_target_road = (agent_y - road.top_y) / 4
                    dist_start_of_target_road = (road.bottom_y - agent_y + 1) / 4
                cars = self.world.road_uid_to_cars_dict[road.uid]
                for car in cars:
                    car_rect = car.get_hitbox()
                    if not is_overlapping_rectangles(car_rect, game_screen_rect): continue # filter out of sight cars
                    visible_car_count += 1
                    [_, _, car_left_x, car_right_x] = car_rect
                    is_left = car_left_x <= agent_x
                    is_right = agent_x <= car_right_x
                    going_right = car.car_details.go_right
                    if (is_left and not going_right) or (is_right and going_right): continue # filter safe cars
                    car_top_row, car_bottom_row = min(car.rows), max(car.rows)
                    if car_top_row < agent_y:
                        dangerous_car_counts[0] += 1 # dangerous car in the front
                    if agent_y < car_bottom_row:
                        dangerous_car_counts[1] += 1 # dangerous car in the back
                    # if is_left:
                    #     dangerous_car_counts[2] += 1 # dangerous car on the left
                    # if is_right:
                    #     dangerous_car_counts[3] += 1 # dangerous car on the right
                # NOTE: each road can have maximum of 6 cars
                visible_car_count /= 6
                dangerous_car_counts = [cnt / 6 for cnt in dangerous_car_counts]
            obs += [crosswalk_visible, penalty, dist_end_of_target_road, dist_start_of_target_road, visible_car_count]
            obs += dangerous_car_counts
            # print("crosswalk_visible:", crosswalk_visible)
            # print("penalty:", penalty)
            # print("dist_end_of_target_road:", dist_end_of_target_road)
            # print("dist_start_of_target_road:", dist_start_of_target_road)
            # print("visible_car_count:", visible_car_count)
            # print("dangerous_car_counts:", dangerous_car_counts)
        assert len(obs) == 20

        # Distance from crosswalks
        if crosswalk_info:
            crosswalk_closeness_list = []
            for i in range(len(self.world.roads.elements), 0, -1):
                road = self.world.roads.elements[i-1]
                if road.top_y > agent_y: continue # already crossed
                if road.bottom_y < grid_top_y: break # out of sight
                if road.crosswalk is None: continue

                crosswalk_closeness = [0, 0, 0] # up, right, left
                if road.crosswalk.top_row <= agent_y <= road.crosswalk.end_row:
                    crosswalk_closeness[0] = 1
                elif agent_y >= road.crosswalk.end_row:
                    if road.crosswalk.end_row >= grid_top_y:
                        crosswalk_closeness[0] = (y_top_visible_distance - (agent_y - road.crosswalk.end_row)) / (y_top_visible_distance)

                if road.crosswalk.left_end <= agent_x <= road.crosswalk.right_end:
                    crosswalk_closeness[1] = 1
                    crosswalk_closeness[2] = 1
                elif agent_x < road.crosswalk.left_end:
                    if road.crosswalk.left_end <= grid_x_right_end:
                        crosswalk_closeness[1] = (x_visible_distance + 1 - (road.crosswalk.left_end - agent_x)) / (x_visible_distance + 1)
                elif road.crosswalk.right_end < agent_x:
                    if grid_x_left_end <= road.crosswalk.right_end:
                        crosswalk_closeness[2] = (x_visible_distance + 1 - (agent_x - road.crosswalk.right_end)) / (x_visible_distance + 1)

                crosswalk_closeness_list += crosswalk_closeness
                if len(crosswalk_closeness_list) == 3 * 3: break

            while len(crosswalk_closeness_list) < 3 * 3:
                crosswalk_closeness_list += [0.0, 0.0, 0.0] # fill up defaults
            obs += crosswalk_closeness_list
            # for i in range(3):
            #     print(f"crosswalk {i}")
            #     print("up closeness to crosswalk:",  crosswalk_closeness_list[3*i])
            #     print("right closeness to crosswalk:",  crosswalk_closeness_list[3*i + 1])
            #     print("left closeness to crosswalk:",  crosswalk_closeness_list[3*i + 2])
        assert len(obs) == 29

        # Cars
        if cars_info:
            # car_in_adjecent_tiles = []
            # for i in range(8):
            #     # NW, N, NE, E, SE, S, SW, W
            #     dx = [-1, 0, 1, 1, 1, 0, -1, -1] 
            #     dy = [-1, -1, -1, 0, 1, 1, 1, 0]
            #     grid_x = agent_x + dx[i]
            #     grid_y = agent_y + dy[i]
            #     tile_rect = [grid_y - 0.5, grid_y + 0.5, grid_x - 0.5, grid_x + 0.5]
            #     car_in_tile = False
            #     if grid_y in self.world.row_to_cars_dict:
            #         for car in self.world.row_to_cars_dict[grid_y]:
            #             if is_overlapping_rectangles(tile_rect, car.get_hitbox()):
            #                 car_in_tile = True
            #                 break
            #     car_in_adjecent_tiles.append(1.0 if car_in_tile else 0.0)
            # obs += car_in_adjecent_tiles
            # print("car_in_adjecent_tiles up:", car_in_adjecent_tiles[1])
            # print("car_in_adjecent_tiles down:", car_in_adjecent_tiles[5])
            # print("car_in_adjecent_tiles right:", car_in_adjecent_tiles[3])
            # print("car_in_adjecent_tiles left:", car_in_adjecent_tiles[7])
            dangerous_car_closeness_list = []
            dangerous_car_speed_list = []
            # only consider until 4 rows ahead of agent since maximum road size is 4
            for grid_y in range(agent_y - 4, grid_bottom_y + 1):
                left_cars = []
                right_cars = []
                # only consider the roads near the agent
                is_prev_road = prev_road is not None and prev_road.top_y <= grid_y <= prev_road.bottom_y
                is_target_road = target_road is not None and target_road.top_y <= grid_y <= target_road.bottom_y
                if is_prev_road or is_target_road:
                    for car in self.world.row_to_cars_dict[grid_y]:
                        car_left_x, car_right_x = car.get_cur_x_pos()
                        if car_right_x < visible_x_left_end: continue # filter out of sight
                        if visible_x_right_end < car_left_x: continue # filter out of sight
                        car_speed = abs(car.default_speed) if car.is_moving else 0
                        if car_right_x < agent_left_x:
                            # car on the left side of the agent
                            if not car.car_details.go_right: pass # moving away
                            left_car_closeness = (left_visible_dist - (agent_left_x - car_right_x)) / left_visible_dist
                            left_cars.append((left_car_closeness, car_speed))
                        elif agent_right_x < car_left_x:
                            # car on the right side of the agent
                            if car.car_details.go_right: pass # moving away
                            right_car_closeness = (right_visible_dist - (car_left_x - agent_right_x)) / right_visible_dist
                            right_cars.append((right_car_closeness, car_speed))
                        else:
                            # car in the same column as the agent
                            left_car_closeness, right_car_closeness = 1.0, 1.0
                            left_cars.append((left_car_closeness, car_speed))
                            right_cars.append((right_car_closeness, car_speed))
                left_cars.sort(key=lambda x: x[0], reverse=True)
                right_cars.sort(key=lambda x: x[0], reverse=True)
                while len(left_cars) < 2:
                    left_cars.append((0.0, 0.0))
                while len(right_cars) < 2:
                    right_cars.append((0.0, 0.0))
                # consider the distance/closeness of 2 cars on each side 
                dangerous_car_closeness_list += [left_cars[0][0], left_cars[1][0], right_cars[0][0], right_cars[1][0]]
                # only consider the speed of the nearest 1 car on each side
                dangerous_car_speed_list += [left_cars[0][1] / self.max_car_speed, right_cars[0][1] / self.max_car_speed]
                # print(f"car left [{grid_y}]:", left_cars)
                # print(f"car right [{grid_y}]:", right_cars)
            obs += dangerous_car_closeness_list
            obs += dangerous_car_speed_list
        assert len(obs) == 65
        assert max(obs) <= 1.0 and min(obs) >= 0.0
        self.obs = np.array(obs)

    def _update_info(self):
        agent_x, agent_y = self.world.agent.get_cur_location_grid()
        grid_y_start = agent_y - self.camera_height//2 - 2
        if self.gamescreen_width_fixed:
            grid_x_start = self.world.agent.init_pos[0] - self.camera_width//2
        else:
            grid_x_start = agent_x - self.camera_width//2
        visible_x_start = grid_x_start - 0.5
        visible_x_end = visible_x_start + self.camera_width

        if self.info is None:
            road_infos = []
            for road in self.world.roads.elements:
                road_info = {
                    "uid": road.uid,
                    "top_y": road.top_y,
                    "bottom_y": road.bottom_y,
                    "going_right": road.going_right,
                    "car_color": str(road.car_color_type),
                    "penalty": road.risk_detail.penalty.value,
                    "crosswalk_col": int(road.crosswalk.col) if road.crosswalk is not None else -1,
                    "going_right": road.going_right,
                }
                road_infos.append(road_info)
            car_infos = []
            for car in self.world.cars.elements:
                start_row, end_row = car.rows[0], car.rows[-1]
                car_info = {
                    "uid": car.uid,
                    "start_row": start_row,
                    "end_row": end_row,
                    "go_right": car.car_details.go_right,
                    "car_grid_width": car.car_details.car_grid_width,
                    "car_grid_height": car.car_details.car_grid_height,
                    "car_type": car.car_details.car_type,
                    "color": str(car.car_details.color),
                    "penalty": car.car_details.penalty.value,
                }
                car_infos.append(car_info)
            self.info = { "road_metadata": road_infos, "car_metadata": car_infos }

        # play_infos
        is_dead = self.world.agent.is_dead
        terminated = self.game_over
        activated_crosswalk_uid = self.world.roads.activated_crosswalk_uid
        time_left = self.time_left
        real_time_step_passed = self.real_time_step_passed
        game_end_extra_score = self.game_end_extra_score
        cur_episode_score = self.cur_rewards
        total_score = self._total_rewards()
        self.info["play_infos"] = [agent_x, agent_y, is_dead, terminated,
                                   activated_crosswalk_uid, time_left, real_time_step_passed,
                                   game_end_extra_score, cur_episode_score, total_score, self.cur_seed]
        # car_infos
        cars = []
        car_uid_set = set()
        for i in range(self.camera_height):
            grid_y = grid_y_start + i
            if grid_y not in self.world.row_to_cars_dict: continue # no car
            for car in self.world.row_to_cars_dict[grid_y]:
                if car.uid in car_uid_set: continue
                car_uid_set.add(car.uid)
                car_left_x, car_right_x = car.get_cur_x_pos()
                if car_right_x < visible_x_start: continue # out of sight
                if visible_x_end < car_left_x: continue # out of sight
                left_x, right_x = car.get_cur_x_pos()
                top_row, bottom_row = car.rows[0], car.rows[-1]
                cars.append([car.uid, car.cur_location[0], car.cur_location[1],
                            left_x, right_x, top_row, bottom_row,
                            car.default_speed, 1 if car.is_moving else 0])
        self.info["cars"] = cars

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
