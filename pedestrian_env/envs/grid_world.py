from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from pedestrian_env.envs.game_object import Car

class Actions(Enum):
    nothing = 0
    up = 1
    down = 2
    right = 3
    left = 4

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, title="Pedestrian Task", render_mode=None, size=5, tick_on_render=False, steps_per_second = 5):
        if size < 5: raise Exception("size can not be less than 5")
        self.title = title
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.tick_on_render = tick_on_render
        self.steps_per_second = steps_per_second
        self.metadata["render_fps"] = steps_per_second * 5 # render multiple times between each step
        self.pix_square_size = (self.window_size / self.size) # The size of a single grid square in pixels


        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
            }
        )

        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.nothing: np.array([0, 0]),
            Actions.up: np.array([0, -1]),
            Actions.down: np.array([0, 1]),
            Actions.right: np.array([1, 0]),
            Actions.left: np.array([-1, 0]),
        }

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

        self._agent_location = None
        self._target_locations = None
        self.cars = None

    def _get_obs(self):
        return {"agent": self._agent_location, "targets": self._target_locations}

    def _get_info(self):
        return {
            "distance": max([np.linalg.norm(self._agent_location - target, ord=1) for target in self._target_locations])
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([int(self.size / 2), self.size - 1], dtype=int)

        self._target_locations = []
        for i in range(3):
            self._target_locations.append(np.array([int(self.size / 2) + i - 1, 0], dtype=int))

        extra_target_location = self._agent_location
        while np.array_equal(extra_target_location, self._agent_location):
            extra_target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_locations.append(extra_target_location)

        self.cars = []
        for i in [2,4]:
            car_type_seed = self.np_random.integers(0, 11, size=1, dtype=int)[0]
            self.cars.append(Car(0, self.size - i, self.pix_square_size, car_type_seed = car_type_seed))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        for car in self.cars:
            car.move(self.size)

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        if self.has_collided():
            terminated = 1
            reward = -10
        else:
            # An episode is done iff the agent has reached the target
            terminated = 0
            for target_location in self._target_locations:
                terminated += np.array_equal(self._agent_location, target_location)
            reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def has_collided(self):
        for car in self.cars:
            if np.array_equal(self._agent_location, [car.x, car.y]):
                return True
        return False

    def render(self):
        if self.render_mode is None: return
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption(self.title)
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # First we draw the targets
        for target_location in self._target_locations:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    self.pix_square_size * target_location,
                    (self.pix_square_size, self.pix_square_size),
                ),
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * self.pix_square_size,
            self.pix_square_size / 3,
        )

        for car in self.cars:
            car.render(canvas)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            if self.tick_on_render:
                self.clock_tick()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def clock_tick(self):
        return self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
