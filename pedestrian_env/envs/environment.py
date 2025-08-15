import gymnasium as gym

import pygame
import numpy as np

from pedestrian_env.envs.world import World
from pedestrian_env.envs.game_object import Car, load_player_images
from pedestrian_env.envs.car_details import Penalty, get_max_car_grid_width
from pedestrian_env.envs.action import Action, ACTION_DURATION

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
                 height=20,
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
                 debug=False,
                 render_sprite=False):
        if width < 12: raise Exception("minimum width is 13")
        if height < 5: raise Exception("minimum height is 5")
        if episode_duration_sec < 10: raise Exception("minimum episode_duration_sec is 10")
        self.title = title
        self.map_grid_width = width
        self.map_grid_height = height + 1 # add starting lane
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
        assert fixed_episode_seed_range is None or (len(fixed_episode_seed_range) == 2 and fixed_episode_seed_range[0] <= fixed_episode_seed_range[1])
        self.fixed_episode_seed_range = fixed_episode_seed_range
        if fixed_episode_seed_range is not None:
            self.fixed_episode_seed = fixed_episode_seed_range[0]
        self.debug = debug
        self.render_sprite = render_sprite
        self.player_images = load_player_images(self.pix_square_size) if render_sprite else None
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
        self.elapsed = 0
        self.game_over = False
        self.game_end_extra_score = 0
        self.obs = None
        self.info = None
        self.prev_step_state = None

        self.max_car_penalty = Penalty.HIGH.value
        self.max_car_speed = max(Car.CAR_SPEEDS)
        agent_x_buffer = 2 + int(max(get_max_car_grid_width(), self.camera_width)/2)
        agent_min_x = (agent_x_buffer)
        agent_max_x = self.map_grid_width - 1 - agent_x_buffer
        self.agent_move_range = (agent_min_x, agent_max_x)

        # action
        self.action_space = gym.spaces.Discrete(5)
        # obs
        self._define_observation_space()

    def reset(self, seed=None, options=None):
        # set seed at `self.np_random`
        if seed is None and self.fixed_episode_seed_range is not None:
            super().reset(seed=self.fixed_episode_seed)
            if self.fixed_episode_seed >= self.fixed_episode_seed_range[1]:
                self.fixed_episode_seed = self.fixed_episode_seed_range[0]
            else:
                self.fixed_episode_seed += 1
            self.fixed_episode_seed = 1 + self.fixed_episode_seed_range[0]
        else:
            super().reset(seed=seed)
        self.elapsed = 0

        self.best_rewards = max(self.best_rewards, self.cur_rewards)
        self.prev_rewards += self.cur_rewards
        self.cur_rewards = 0
        self.time_left = self.GAME_TIME_MS
        self.game_over = False
        self.game_end_extra_score = 0

        self.world = World(self.agent_move_range, self.map_grid_width, self.map_grid_height, self.camera_width, self.pix_square_size, self.steps_per_second, self.np_random, self.debug, self.render_sprite, self.player_images)

        if self.render_mode == "human":
            self.render()

        if self.debug:
            print(self.world.cars)

        self.obs = None
        self.info = None
        self.prev_step_state = dict()
        self._update_state_snapshot()
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
        if not self.realtime:
            # interacting with the environment using RL algorithm
            self.world.agent.update_target(action)
            for _ in range(ACTION_DURATION[action]):
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
                    if cur_count >= ACTION_DURATION[action]: break
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
        if self.world.target_lane_reached():
            self.game_over = True
            game_end_extra_score = self._get_time_left_sec()
            cumulative_reward += game_end_extra_score * self.BONUS_SCORE_PER_SEC
            self.game_end_extra_score = game_end_extra_score * self.BONUS_SCORE_PER_SEC

        if self.game_over and self.realtime:
            self._render_game_over()
        
        cumulative_reward += self.world.calculate_cum_crossing_rewards()
        self.cur_rewards += cumulative_reward
        self._update_state_snapshot()
        return self.obs, cumulative_reward, self.game_over, False, self.info
    
    def _check_gameover(self):
        reward, terminated = 0, False
        agent_dead, death_penalty = self.world.cars.has_hit_agent()
        if agent_dead:
            terminated = True
            self.world.agent.set_dead()
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
                (agent_min_x, agent_max_x) = self.agent_move_range
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
                agent_dead, _ = self.world.cars.has_hit_agent()
                if agent_dead:
                    self.world.agent.set_dead()
            self.apply_time_and_render(dt)

    def _define_observation_space(self):
        self.channel_count = 11
        lower_bounds = [0, 0, 0, 0, 0, 
                        0, 0, 0, -1, 0,0]
        upper_bounds = [1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1]
        if self.gamescreen_width_fixed:
            self.channel_count += 1
            lower_bounds.append(0)
            upper_bounds.append(1)

        low_hwc = np.full((self.camera_height, self.camera_width, self.channel_count), lower_bounds)
        high_hwc = np.full((self.camera_height, self.camera_width, self.channel_count), upper_bounds)

        low_chw = np.transpose(low_hwc, (2, 0, 1))
        high_chw = np.transpose(high_hwc, (2, 0, 1))
        self.observation_space = gym.spaces.Box(low=low_chw, high=high_chw, dtype=np.float32)

    def _update_state_snapshot(self):
        self._update_obs()
        self._update_info()

    def _update_obs(self):
        """
        Observation (normalized)
        - structure: (C, H, W) = (channels, y, x)

        Channel 0: Danger zone
        - 0: safe zone (or unreachable)
        - 1: danger zone (includes crosswalks because cars on crosswalks can still kill the agent)

        Channel 1: Crosswalk
        - 0: not crosswalk
        - 1: crosswalk

        Channel 2: Crosswalk Activation (Crosswalk x Agent)
        - 0: not activated crosswalk
        - 1: activated crosswalk with agent

        Channel 3: Reachable tile
        - 0: unreachable
        - 1: reachable or target area

        Channel 4: Reward tile
        - 0: no reward
        - 1: give reward on reaching the tile with UP action (same amount as penalty on leaving the tile with DOWN action)
        - NOTE: should be removed in AIRL reward_net to prevent Reward leakage

        Channel 5: Car tile (soft mask)
        - 0    : no car on tile
        - 0 ~ 1: proportion of tile covered by car

        Channel 6: Car ingress delta
        - 0 ~ 1: how much a car moved into the tile (compared to previous observation)
        - 0    : no incoming movement

        Channel 7: Car egress delta
        - 0 ~ 1: how much a car moved out of the tile (compared to previous observation)
        - 0    : no outgoing movement

        Channel 8: Car speed
        - 0     : no car or stopped
        - -1 ~ 0: car going left speed
        - 0 ~ +1: car going right speed

        Channel 9: Car penalty
        - 0    : no car in the tile
        - 0 ~ 1: penalty of the existing car (normalized)

        Channel 10: Play time left
        - 0 ~ 1: time remaining (normalized)
        - 0    : game over (time over, death, early finish)

        Extra Channel: Agent position (gamescreen_width_fixed == True)
        - 0: not agent
        - 1: agent
        """
        agent_x, agent_y = self.world.agent.get_cur_location_grid()
        grid_y_start = agent_y - self.camera_height//2 - 2
        if self.gamescreen_width_fixed:
            grid_x_start = self.world.agent.init_pos[0] - self.camera_width//2
        else:
            grid_x_start = agent_x - self.camera_width//2
        visible_x_start = grid_x_start - 0.5
        visible_x_end = visible_x_start + self.camera_width

        # calculate car info (include cars out of sight with buffer of grid size 1)
        prev_left_covered_dict = self.prev_step_state["left_covered"] if "left_covered" in self.prev_step_state else dict()
        prev_right_covered_dict = self.prev_step_state["right_covered"]  if "right_covered" in self.prev_step_state else dict()

        cur_left_covered_dict = dict()
        cur_fully_covered_set = set()
        cur_right_covered_dict = dict()
        cur_car_info_dict = dict()
        for y in range(self.camera_height + 2):
            grid_y = grid_y_start + y - 1
            if grid_y not in self.world.row_to_cars_dict: continue # no car
            for car in self.world.row_to_cars_dict[grid_y]:
                car_left_x, car_right_x = car.get_cur_x_pos()
                if car_right_x < visible_x_start - 1: continue
                if visible_x_end + 1 < car_left_x: continue

                car_left_x_adjusted = car_left_x + 0.5
                car_right_x_adjusted = car_right_x + 0.5

                grix_left_end_x = int(car_left_x_adjusted)
                cur_right_covered_dict[(grid_y, grix_left_end_x)] = grix_left_end_x + 1 - car_left_x_adjusted # right part of the tile partially covered
    
                grix_right_end_x = int(car_right_x_adjusted)
                cur_left_covered_dict[(grid_y, grix_right_end_x)] = car_right_x_adjusted - grix_right_end_x # left part of the tile partially covered
    
                for grid_x in range(grix_left_end_x + 1, grix_right_end_x):
                    cur_fully_covered_set.add((grid_y, grid_x)) # tile fully covered by a single car
                for grid_x in range(int(max(visible_x_start - 0.5, grix_left_end_x)), int(min(visible_x_end + 0.5, grix_right_end_x)+1)):
                    cur_speed = car.get_cur_speed()
                    penalty = car.car_detail.penalty.value
                    if (grid_y, grid_x) not in cur_car_info_dict:
                        cur_car_info_dict[(grid_y, grid_x)] = [car.default_speed > 0, cur_speed, penalty]
                    else:
                        if abs(cur_speed) > abs(cur_car_info_dict[(grid_y, grid_x)][1]):
                            cur_car_info_dict[(grid_y, grid_x)][1] = cur_speed
                        cur_car_info_dict[(grid_y, grid_x)][2] = max(penalty, cur_car_info_dict[(grid_y, grid_x)][2])

        # fill up obs
        obs = np.zeros((self.channel_count, self.camera_height, self.camera_width), dtype=np.float32)
        for y in range(self.camera_height):
            grid_y = grid_y_start + y
            # fill up visible, but not reachable area to encourage reaching the end of the map
            if grid_y < 0:
                # Channel 3: Reachable tile
                obs[3][y] = 1
                # Channel 4: Reward tile
                obs[4][y] = 1
            # fill up reachable and visible grids
            elif 0 <= grid_y < self.map_grid_height:
                for x in range(self.camera_width):
                    grid_x = grid_x_start + x
                    if (0 <= grid_x < self.map_grid_width):
                        # Channel 0: Danger tile
                        obs[0][y][x] = 1 if self.world.danger_map[grid_y][grid_x] else 0
                        # Channel 1: Crosswalk
                        if self.world.crosswalk_map[grid_y][grid_x]:
                            obs[1][y][x] = 1
                        # Channel 3: Reachable tile
                        obs[3][y][x] = 1 if self.world.reachable_map[grid_y][grid_x] else 0
                        # Channel 4: Reward tile
                        obs[4][y][x] = 1 if self.world.reward_y[grid_y] else 0
                    if agent_x == grid_x and agent_y == grid_y:
                        # Channel 2: Crosswalk Activation (Crosswalk x Agent)
                        if self.world.crosswalk_map[grid_y][grid_x]:
                            obs[2][y][x] = 1
                        # Channel 11: Agent position
                        if self.gamescreen_width_fixed:
                            obs[11][y][x] = 1
                    
                    key = (grid_y, grid_x)
                    if key in cur_car_info_dict:
                        [going_right, cur_speed, penalty] = cur_car_info_dict[key]
                        # Channel 5: Car tile (soft mask)
                        if key in cur_fully_covered_set:
                            obs[5][y][x] = 1 # tile fully covered by a single car
                        elif (key in cur_left_covered_dict and key in cur_right_covered_dict):
                            obs[5][y][x] = 1 # both ends of the tile covered by two cars
                        elif key in cur_left_covered_dict:
                            obs[5][y][x] = cur_left_covered_dict[key] # left part of the tile partially covered
                        elif key in cur_right_covered_dict:
                            obs[5][y][x] = cur_right_covered_dict[key] # right part of the tile partially covered
                    
                        cur_covered, prev_covered = 0, 0
                        ingress, egress = 0, 0
                        if key in cur_left_covered_dict:
                            cur_covered = cur_left_covered_dict[key]
                            if key in prev_left_covered_dict:
                                prev_covered = prev_left_covered_dict[key]
                            if going_right:
                                ingress = cur_covered - prev_covered
                            else:
                                egress = prev_covered - cur_covered
                        if key in cur_right_covered_dict:
                            cur_covered = cur_right_covered_dict[key]
                            if key in prev_right_covered_dict:
                                prev_covered = prev_right_covered_dict[key]
                            if not going_right:
                                ingress = cur_covered - prev_covered
                            else:
                                egress = prev_covered - cur_covered
                        # Channel 6: Car ingress delta
                        obs[6][y][x] = max(0, ingress)
                        # Channel 7: Car egress delta
                        obs[7][y][x] = max(0, egress)
                        # Channel 8: Car speed
                        obs[8][y][x] = cur_speed / self.max_car_speed
                        # Channel 9: Car penalty
                        obs[9][y][x] = penalty / self.max_car_penalty

        self.prev_step_state["left_covered"] = cur_left_covered_dict
        # self.prev_step_state["fully_covered"] = cur_fully_covered_set
        self.prev_step_state["right_covered"] = cur_right_covered_dict
        # Channel 10: Play time left
        if self.game_over or self.time_left - 1000 <= 0:
            obs[10] = 0
        else:
            obs[10] = (self.time_left - 1000) / (self.GAME_TIME_MS  - 1000)
        self.obs = obs

    def _update_info(self):
        """
        Returns:
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
        """
        agent_x, agent_y = self.world.agent.get_cur_location_rounded()
        self.info = {
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
# self.pix_square_size = 50
# self.camera_width_pixel = width * self.pix_square_size
# self.camera_height_pixel = height * self.pix_square_size
# visualizing tile example
# for pos in [(12, 20), (13, 20), (14, 20), (15, 20), (15, 19), (15, 18), (15, 17), (14, 17), (13, 17), (12, 17), (11, 17), (10, 17), (9, 17), (8, 17), (8, 16), (8, 15), (8, 14), (9, 14), (10, 14), (11, 14), (12, 14),
#              (13, 14), (14, 14), (15, 14), (15, 13), (15, 12), (14, 12), (14, 11), (13, 11), (12, 11), (12, 10), (11, 10), (10, 10), (9, 10), (8, 10), (8, 9), (8, 8)]:
#     start_x = pos[0]
#     start_y = pos[1]
#     pygame.draw.rect(canvas, 
#                     (150, 150, 200), 
#                     ((start_x - 0.5) * self.pix_square_size, start_y * self.pix_square_size, self.pix_square_size, self.pix_square_size))

# for x in range(25):
#     start_y = 0
#     pygame.draw.line(
#         canvas,
#         (0, 0, 0),
#         (self.pix_square_size * (x + 0.5), start_y),
#         (self.pix_square_size * (0.5 + x), self.pix_square_size * self.map_grid_height),
#         width=3,
#     )
# for y in range(21):
#     start_x = 0
#     pygame.draw.line(
#         canvas,
#         (0, 0, 0),
#         (start_x, self.pix_square_size * y),
#         (self.map_width, self.pix_square_size * y),
#         width=3,
#     )



