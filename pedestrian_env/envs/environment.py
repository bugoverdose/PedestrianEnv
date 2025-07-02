import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from pedestrian_env.envs.action import ACTION_TO_DELTA
from pedestrian_env.envs.world import World

class PedestrianEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    OFF_SCREEN_BLACK_COLOR = (0, 0, 0)
    UI_TEXT_WHITE_COLOR = (255, 255, 255)
    AGENT_DEAD_TEXT_COLOR = (255, 0, 0)
    SUCCESS_TEXT_COLOR = (0, 255, 0)
    TIME_OVER_TEXT_COLOR = (0, 0, 255)

    EXTRA_WIDTH = 500
    EXTRA_HEIGHT = 400

    DEATH_PENALTY = 100

    DEFAULT_GAME_TIME = 60_000
    DEFAULT_GAMEOVER_REST_TIME = 3_000

    def __init__(self, title="Pedestrian Task", width=10, height=10, camera_size=5, render_mode=None, tick_on_render=False, steps_per_second = 10, debug=False):
        if width < 5 or height < 5: raise Exception("width or height can not be less than 5")
        self.title = title
        self.map_grid_width = width
        self.map_grid_height = height
        self.tick_on_render = tick_on_render
        self.steps_per_second = steps_per_second
        self.step_ms =  1000 / self.steps_per_second # default: step once every 100ms
        self.metadata["render_fps"] = 60 # NOTE: must render multiple times between each step
        base_window_size = 2048
        self.pix_square_size = (base_window_size / max(self.map_grid_width, self.map_grid_height)) # The size of a single grid square in pixels
        self.camera_size = camera_size * self.pix_square_size
        self.map_width = self.map_grid_width * self.pix_square_size
        self.map_height = self.map_grid_height * self.pix_square_size
        self.debug = debug

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([self.map_grid_width, self.map_grid_height]),
                    dtype=np.float32 # continuous space
                ),
                # TODO: add nearby car info
            }
        )

        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.world = None

        self.prev_rewards = 0 # sum of all the rewards from all the previous episodes
        self.cur_rewards = 0 # reward from the current ongoing episode
        self.GAME_TIME = self.DEFAULT_GAME_TIME + 1_000
        if debug:
            self.GAME_TIME = 11_000
        self.time_left = self.GAME_TIME
        self.game_over = False
        self.game_end_extra_score = 0

    def _total_rewards(self):
        return self.prev_rewards + self.cur_rewards

    def _get_obs(self):
        """
        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
        """
        return {"agent": self.world.agent.cur_location, "cur_rewards": self.cur_rewards, "total_rewards": self._total_rewards()}

    def _get_info(self):
        """
        Returns:
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
        """
        return {
            "is_dead": self.world.agent.is_dead,
            "time_left_ms": self.time_left,
            "game_end_extra_score": self.game_end_extra_score,
        }

    def reset(self, seed=None, options=None):
        # NOTE: following line is needed for self.np_random
        super().reset(seed=seed)

        self.prev_rewards += self.cur_rewards
        self.cur_rewards = 0
        self.time_left = self.GAME_TIME
        self.game_over = False
        self.game_end_extra_score = 0

        self.world = World(self.map_grid_width, self.map_grid_height, self.pix_square_size, self.steps_per_second, self.np_random, self.debug)

        if self.render_mode == "human":
            self.render()

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
        direction = ACTION_TO_DELTA[action]
        # We use `np.clip` to make sure the agent doesn't leave the grid
        self.world.agent.update_target(direction)

        cur_up_rewards = self.world.calculate_up_rewards()
        reward = cur_up_rewards - prev_up_rewards
        if self.world.has_collided():
            terminated = True
            self.world.agent.set_dead()
            reward -= self.DEATH_PENALTY
            self.game_end_extra_score = -self.DEATH_PENALTY
        elif self.get_time_left_sec() <= 0:
            terminated = True
        else:
            # An episode is finished if the agent has reached the target lane
            terminated = self.world.target_lane_reached()
            if terminated:
                game_end_extra_score = self.get_time_left_sec()
                reward += game_end_extra_score
                self.game_end_extra_score = game_end_extra_score
        self.game_over = terminated
        self.cur_rewards += reward
        return self._get_obs(), reward, terminated, False, self._get_info()

    def update_positions(self, dt):
        self.world.update_positions(dt)

    def render(self):
        if self.render_mode is None: return
        total_window_width = self.camera_size + self.EXTRA_WIDTH
        total_window_height = self.camera_size + self.EXTRA_HEIGHT

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption(self.title)
            pygame.display.init()
            self.window = pygame.display.set_mode((total_window_width, total_window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas, rendered_agent_position = self.world.render()

        camera_rect = pygame.Rect(0, 0, self.camera_size, self.camera_size)
        camera_rect.center = rendered_agent_position
        camera_rect.clamp_ip(canvas.get_rect()) # prevent camera from going out-of-bounds

        if self.render_mode == "human":
            left_ui_x = total_window_width * 0.2
            right_ui_x = total_window_width * 0.7
            top_y = total_window_height * 0.15
            bottom_y = total_window_height * 0.85
            extra_score_text_width = 155
            left_game_x = self.EXTRA_WIDTH/2
            top_game_y = self.EXTRA_HEIGHT/2

            # add game screen
            self.window.blit(canvas, (left_game_x, top_game_y), area=camera_rect)

            if self.game_over:
                dark_screen = pygame.Surface((self.world.map_width, self.world.map_height), pygame.SRCALPHA)
                alpha = 128 # 50% transparency
                dark_screen.fill((0, 0, 0, alpha))
                self.window.blit(dark_screen, (left_game_x, top_game_y), area=camera_rect)

                if self.game_end_extra_score < 0:
                    game_status_desc = "YOU DIED"
                    game_status_color = self.AGENT_DEAD_TEXT_COLOR
                elif self.game_end_extra_score > 0:
                    game_status_desc = "SUCCESS"
                    game_status_color = self.SUCCESS_TEXT_COLOR
                else:
                    game_status_desc = "TOO SLOW"
                    game_status_color = self.TIME_OVER_TEXT_COLOR
                self.render_text(total_window_width/2, total_window_height/2, game_status_desc, game_status_color, 128, True)

            # clear and update score area
            bg_rect = pygame.Rect(0 , total_window_height - top_game_y, total_window_width, top_game_y)
            pygame.draw.rect(self.window, self.OFF_SCREEN_BLACK_COLOR, bg_rect)
            # NOTE: show the sum of previous scores as total score (because it's confusing when both cur rewards and total rewards are constantly changing)
            self.render_text(left_ui_x, bottom_y, f"Total Score: {self.prev_rewards}",self.UI_TEXT_WHITE_COLOR, 32)

            cur_score = self.cur_rewards
            if self.game_end_extra_score < 0:
                cur_score += self.DEATH_PENALTY
                self.render_text(right_ui_x + extra_score_text_width, bottom_y+32, f"-{self.DEATH_PENALTY}",(255, 0, 0), 32)
            elif self.game_end_extra_score > 0:
                cur_score -= self.game_end_extra_score
                self.render_text(right_ui_x + extra_score_text_width, bottom_y+32, f"+{self.game_end_extra_score}",(0, 255, 0), 32)
            self.render_text(right_ui_x, bottom_y, f"Current Score: {cur_score}",self.UI_TEXT_WHITE_COLOR, 32)

            # clear and update time left
            bg_rect = pygame.Rect(0 , 0, total_window_width, top_game_y)
            pygame.draw.rect(self.window, self.OFF_SCREEN_BLACK_COLOR, bg_rect)
            self.render_text(right_ui_x, top_y, f"Time Left: {self.get_time_left_sec()}",self.UI_TEXT_WHITE_COLOR, 32)

            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            if self.tick_on_render:
                self.clock_tick()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def render_text(self, x, y, text, color, font_size=32, center=False):
        font = pygame.font.SysFont(None, font_size)
        text_surface = font.render(text, True, color)
        if center:
            text_location = text_surface.get_rect(center=(x, y))
        else:
            text_location = (x, y)
        self.window.blit(text_surface, text_location)

    def render_game_over(self):
        elapsed = 0
        total_elapsed = 0
        while total_elapsed < self.DEFAULT_GAMEOVER_REST_TIME:
            dt = self.clock_tick()
            elapsed += dt
            total_elapsed += dt
            while elapsed >= self.step_ms:
                elapsed -= self.step_ms
                if self.world.has_collided():
                    self.world.agent.set_dead()
            self.update_positions(dt)
            self.render()

    def clock_tick(self):
        return self.clock.tick(self.metadata["render_fps"])

    def get_time_left_sec(self):
        return int(self.time_left/1000)

    def update_time_left(self, elapsed_time):
        if self.game_over: return
        self.time_left = self.GAME_TIME - elapsed_time

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

