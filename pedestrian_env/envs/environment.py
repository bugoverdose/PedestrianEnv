import gymnasium as gym

import pygame
import numpy as np

from pedestrian_env.envs.world import World
from pedestrian_env.envs.game_object import Agent, Car
from pedestrian_env.envs.car_details import get_max_car_grid_width, get_max_panalty
from pedestrian_env.envs.action import Action, ACTION_DURATION
from pedestrian_env.envs.utils import is_overlapping

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
                 width=25, height=20,
                 camera_width=11, camera_height=7,
                 extra_reward_using_crosswalk=False,
                 gamescreen_width_fixed=True,
                 render_mode=None,
                 tick_on_render=False,
                 steps_per_second=10,
                 realtime=False,
                 episode_duration_sec=30,
                 gameover_screen_time=5000,
                 debug=False,
                 render_sprite=False):
        if width < 12: raise Exception("minimum width is 13")
        if height < 5: raise Exception("minimum height is 5")
        if episode_duration_sec < 10: raise Exception("minimum episode_duration_sec is 10")
        self.title = title
        self.map_grid_width = width
        self.map_grid_height = height + 1 # add starting lane
        self.tick_on_render = tick_on_render
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
        self.elapsed = 0
        self.game_over = False
        self.game_end_extra_score = 0

        self.max_car_penalty = get_max_panalty()
        self.max_car_speed = max(Car.CAR_SPEEDS)

        # action
        self.action_space = gym.spaces.Discrete(5)
        # obs
        self._define_observation_space()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # set seed at `self.np_random`
        self.elapsed = 0

        self.best_rewards = max(self.best_rewards, self.cur_rewards)
        self.prev_rewards += self.cur_rewards
        self.cur_rewards = 0
        self.time_left = self.GAME_TIME_MS
        self.game_over = False
        self.game_end_extra_score = 0

        agent_x_buffer = 2 + int(max(get_max_car_grid_width(), self.camera_width)/2)
        agent_min_x = (agent_x_buffer)
        agent_max_x = self.map_grid_width - 1 - agent_x_buffer
        self.world = World((agent_min_x, agent_max_x), self.map_grid_width, self.map_grid_height, self.camera_width, self.pix_square_size, self.steps_per_second, self.np_random, self.debug, self.render_sprite)

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
        return self._get_obs(), cumulative_reward, self.game_over, False, self._get_info()
    
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
            camera_rect.center = (self.world.agent.init_pos[0] * self.pix_square_size, rendered_agent_position[1])
        else:
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
        self.channel_count = 7
        lower_bounds = (0, 0, 0, 0, -1, 0, 0)
        upper_bounds = (1, 1, 1, 1,  1, 1, 1)
        if self.gamescreen_width_fixed:
            self.channel_count += 1
            lower_bounds = (0, 0, 0, 0, -1, 0, 0, 0)
            upper_bounds = (1, 1, 1, 1,  1, 1, 1, 1)

        low_hwc = np.full((self.camera_height, self.camera_width, self.channel_count), lower_bounds)
        high_hwc = np.full((self.camera_height, self.camera_width, self.channel_count), upper_bounds)

        low_chw = np.transpose(low_hwc, (2, 0, 1))
        high_chw = np.transpose(high_hwc, (2, 0, 1))
        self.observation_space = gym.spaces.Box(low=low_chw, high=high_chw, dtype=np.float32)

    def _get_obs(self):
        """
        Observation (normalized)
        - structure: (C, H, W) = (channels, y, x)

        Channel 0: Danger tile
        - 0: safe zone (or unreachable)
        - 1: danger zone

        Channel 1: Closeness to crosswalk
        - 0.0: crosswalk not visible (or unreachable)
        - 0.0 ~ 1.0: closeness to the crosswalk
        - 1.0: crosswalk

        Channel 2: Reachable tile
        - 0: unreachable
        - 1: reachable or target area

        Channel 3: Car penalty
        - 0 : no car in the tile
        - 0.1, 0.5, 1.0 : penalty of the existing car (raw range: 100, 500, 1000)

        Channel 4: Car speed
        - 0: stopped car or no car in the tile
        - -1.0 ~ 0: going left (raw range: -4.5 ~ -3.0)
        - 0 ~ 1.0: going right (raw range: 3.0 ~ 4.5)

        Channel 5: Risk level
        - 0.0: no car visible or moving away from the agent (safe)
        - 0 ~ 1: how close the agent is from the car
        - 1.0: in front of the agent or hit the agent (maximum danger)

        Channel 6: Play time left
        - 0: game over (time over, death, early finish)
        - 0 ~ 1: time left / maximum episode time
        - 1: episode initialized

        [if gamescreen_width_fixed == True]
        Channel 7: Agent position
        - 0: agent
        - 1: not agent
        """
        agent_x, agent_y = self.world.agent.get_cur_location_grid()
        agent_left_x, agent_right_x = agent_x - Agent.RADIUS, agent_x + Agent.RADIUS
        max_x_dist_from_agent = self.camera_width/2 - Agent.RADIUS
        grid_y_start = agent_y - self.camera_height//2
        if self.gamescreen_width_fixed:
            grid_x_start = self.world.agent.init_pos[0] - self.camera_width//2
        else:
            grid_x_start = agent_x - self.camera_width//2
        visible_x_start = grid_x_start - 0.5
        visible_x_end = visible_x_start + self.camera_width
        obs = np.zeros((self.channel_count, self.camera_height, self.camera_width), dtype=np.float32)
        crosswalk_pos_set = set()
        for y in range(self.camera_height):
            grid_y = grid_y_start + y
            for x in range(self.camera_width):
                grid_x = grid_x_start + x
                if (0 <= grid_x < self.map_grid_width) and (0 <= grid_y < self.map_grid_height):
                    # Channel 0: Danger tile
                    obs[0][y][x] = 1 if self.world.danger_map[grid_y][grid_x] else 0
                    # Channel 1: Closeness to crosswalk
                    if self.world.crosswalk_map[grid_y][grid_x]:
                        crosswalk_pos_set.add((y, x))
                        obs[1][y][x] = 1.0
                    # Channel 2: Reachable tile
                    obs[2][y][x] = 1 if self.world.reachable_map[grid_y][grid_x] else 0
                if self.gamescreen_width_fixed:
                    # Channel 7: Agent position
                    if agent_x == grid_x and agent_y == grid_y:
                        obs[7][y][x] = 1.0
                elif grid_y < 0:
                    # Channel 2: Reachable tile
                    obs[2][y][x] = 1 # area after the target is actually not reachable because the game will be stopped, but make it look like safe zone
            if grid_y not in self.world.row_to_cars_dict: continue # no car
            for car in self.world.row_to_cars_dict[grid_y]:
                left_x, right_x = car.get_cur_x_pos()
                if right_x < visible_x_start: continue # out of sight
                if visible_x_end < left_x: continue # out of sight
                for x in range(self.camera_width):
                    block_left = x + visible_x_start
                    block_right = x + visible_x_start + 1
                    if not is_overlapping(left_x, right_x, block_left, block_right): continue
                    # Channel 3: Car penalty
                    obs[3][y][x] = car.car_detail.penalty / self.max_car_penalty
                    # Channel 4: Car speed
                    obs[4][y][x] = car.get_cur_speed() / self.max_car_speed
                    # Channel 5: Risk level
                    if x == self.camera_width//2: # same column as the agent
                        obs[5][y][x] = 1
                    elif x < self.camera_width//2: # left from the agent
                        if car.default_speed < 0: continue # going away
                        car_right_block_end = right_x if block_left <= right_x < block_right else block_right
                        obs[5][y][x] = max(obs[5][y][x], 1 - ((agent_left_x - car_right_block_end) / max_x_dist_from_agent))
                    else: # right from the agent
                        if car.default_speed > 0: continue # going away
                        car_left_block_end = left_x if block_left <= left_x < block_right else block_left
                        obs[5][y][x] = max(obs[5][y][x], 1 - ((car_left_block_end - agent_right_x) / max_x_dist_from_agent))

        # Channel 1: Closeness to crosswalk
        for i in range(1, 10):
            dist = 1.0 - 0.1*i
            new_set = set()
            for (y, x) in crosswalk_pos_set:
                for j in [1, -1]:
                    adj_x = x + j
                    if adj_x < 0 or adj_x >= self.camera_width: continue
                    if obs[1][y][adj_x] > dist: continue
                    if obs[0][y][adj_x] > 0: continue # ignore danger zone
                    obs[1][y][adj_x] = dist
                    new_set.add((y, adj_x))
            crosswalk_pos_set = new_set
            if len(crosswalk_pos_set) == 0: break

        # Channel 6: Play time left
        if self.game_over or self.time_left - 1000 <= 0:
            obs[6] = 0 # game over
        elif self.time_left >= self.GAME_TIME_MS:
            obs[6] = 1 # episode initialized
        else:
            obs[6] = (self.time_left - 1000) / (self.GAME_TIME_MS  - 1000)
 
        return obs

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
