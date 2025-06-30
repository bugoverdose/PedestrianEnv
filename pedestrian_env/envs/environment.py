import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from pedestrian_env.envs.action import Action, ACTION_TO_DELTA
from pedestrian_env.envs.world import World


class PedestrianEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, title="Pedestrian Task", width=10, height=10, camera_size=10, render_mode=None, tick_on_render=False, steps_per_second = 5):
        if width < 5 or height < 5: raise Exception("width or height can not be less than 5")
        self.title = title
        self.width = width  # max width of the world
        self.height = height  # max height of the world
        self.tick_on_render = tick_on_render
        self.steps_per_second = steps_per_second
        self.metadata["render_fps"] = 60 # NOTE: must render multiple times between each step
        base_window_size = 1024
        self.pix_square_size = (base_window_size / max(self.width, self.height)) # The size of a single grid square in pixels
        self.camera_size = camera_size * self.pix_square_size
        self.map_width = self.width * self.pix_square_size
        self.map_height = self.height * self.pix_square_size

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0, 0]), high=np.array([self.width - 1, self.height - 1]), dtype=np.int32),
                "target": spaces.Box(low=np.array([0, 0]), high=np.array([self.width - 1, self.height - 1]), dtype=np.int32),
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

    def _total_rewards(self):
        return self.prev_rewards + self.cur_rewards

    def _get_obs(self):
        """
        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
        """
        return {"agent": self.world.agent_location, "cur_rewards": self.cur_rewards, "total_rewards": self._total_rewards()}

    def _get_info(self):
        """
        Returns:
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
        """
        return {}

    def reset(self, seed=None, options=None):
        # NOTE: following line is needed for self.np_random
        super().reset(seed=seed)

        self.prev_rewards += self.cur_rewards
        self.cur_rewards = 0

        self.world = World(self.width, self.height, self.pix_square_size, self.np_random)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

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
        for car in self.world.cars:
            car.move(self.width)

        prev_up_rewards = self.world.calculate_up_rewards()
        # Map the action to the direction we walk in
        direction = ACTION_TO_DELTA[action]
        # We use `np.clip` to make sure we don't leave the grid
        self.world.agent_target_location = np.clip(self.world.agent_location + direction, [0, 0], [self.width - 1, self.height - 1])

        cur_up_rewards = self.world.calculate_up_rewards()
        reward = cur_up_rewards - prev_up_rewards
        if self.world.has_collided():
            terminated = True
            reward -= 1000
        else:
            # An episode is finished if the agent has reached the target lane
            cur_y = self.world.get_agent_cur_y()
            terminated = cur_y == 0
            if terminated:
                reward += 100

        self.cur_rewards += reward

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def update_positions(self, dt):
        self.world.update_positions(dt)

    def render(self):
        if self.render_mode is None: return
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption(self.title)
            pygame.display.init()
            self.window = pygame.display.set_mode((self.camera_size, self.camera_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.map_width, self.map_height))
        canvas.fill((255, 255, 255))

        # Now we draw the agent
        agent_x = (self.world.agent_location[0] + 0.5) * self.pix_square_size
        agent_y = (self.world.agent_location[1] + 0.5) * self.pix_square_size
        agent_position = (agent_x, agent_y) # agent_position = (self.world.agent_location + 0.5) * self.pix_square_size
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            agent_position,
            self.pix_square_size / 3,
        )

        for car in self.world.cars:
            car.render(canvas)

        # Finally, add some gridlines
        for x in range(self.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, self.pix_square_size * x),
                (self.map_width, self.pix_square_size * x),
                width=3,
            )
        for x in range(self.width + 1):
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.map_height),
                width=3,
            )

        camera_rect = pygame.Rect(0, 0, self.camera_size, self.camera_size)
        camera_rect.center = agent_position
        camera_rect.clamp_ip(canvas.get_rect()) # prevent camera from going out-of-bounds

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, (0, 0), area=camera_rect)

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            if self.tick_on_render:
                self.clock_tick()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def clock_tick(self):
        return self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
