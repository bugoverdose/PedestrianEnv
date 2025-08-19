import numpy as np

import gymnasium as gym

from imitation.data.types import TrajectoryWithRew

# NOTE: Variable Horizon Environments Considered Harmful: https://imitation.readthedocs.io/en/latest/main-concepts/variable_horizon.html
class FixedHorizonAbsorbIndicator(gym.Wrapper):
    """
    add Absorbing Indicator Channel (ongoing=0, absorbing=1)

    add additional transitions until reaching `max_step`
    - termination ~ max_step-1 : terminated=False, truncated=False
    - max_step                 : terminated=False, truncated=True

    add info["absorbing"]
    - ~ termination          : info["absorbing"] = False
    - termination ~ max_step : info["absorbing"] = True
    """
    def __init__(self, env, max_step: int):
        super().__init__(env)
        self.max_step = int(max_step)
        self.cur_step = 0
        self.is_absorbing = False

        assert isinstance(env.observation_space, gym.spaces.Box)
        C, Hh, Ww = env.observation_space.shape

        # Box(C,H,W) -> Box(C+1,H,W)
        low  = env.observation_space.low
        high = env.observation_space.high
        ind_low  = np.zeros((1, Hh, Ww), dtype=low.dtype)
        ind_high = np.ones((1, Hh, Ww), dtype=high.dtype)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, ind_low], axis=0),
            high=np.concatenate([high, ind_high], axis=0),
            dtype=env.observation_space.dtype
        )
        self._absorbing_obs = np.zeros((C+1, Hh, Ww), dtype=self.observation_space.dtype)
        self._absorbing_obs[-1, :, :] = 1.0

    def update_last_ongoing_info(self, info):
        info = {} if info is None else dict(info)
        info["absorbing"] = False
        self.last_info = info

    def reset(self, **kw):
        self.cur_step = 0
        self.is_absorbing = False
        obs, info = self.env.reset(**kw)
        self.update_last_ongoing_info(info)
        return _append_absorb_indicator(obs, 0.0), self.last_info

    def step(self, action):
        self.cur_step += 1
        if not self.is_absorbing:
            obs, rew, terminated, truncated, info = self.env.step(action)
            obs = _append_absorb_indicator(obs, 0.0)
            self.update_last_ongoing_info(info)
            if terminated or truncated:
                self.is_absorbing = True # start absorbing after current transition
        else:
            obs = self._absorbing_obs
            rew = 0.0
            if not self.last_info["absorbing"]:
                self.last_info["absorbing"] = True
        return obs, rew, False, self.cur_step >= self.max_step, self.last_info

NO_OP_ACTION = 0 # == Action.NOTHING.value

def create_fixed_horizon_TrajectoryWithRew(
        observations, # shape: (T+1,), dtype: float32
        actions,      # shape: (T,), dtype: int64
        rewards,      # shape: (T,), dtype: float32
        infos,        # shape: (T,)
        max_step
    ):
    T = len(actions)
    # add Absorbing Indicator Channel 
    obs = np.stack([_append_absorb_indicator(obs, 0.0) for obs in observations], axis=0)
    pad_T = max_step - T
    if pad_T > 0:
        C,H,W = obs.shape[1:]
        absorb = np.zeros((C,H,W), dtype=np.float32)
        absorb[-1] = 1.0
        obs_pad = np.stack([absorb]*pad_T, axis=0)   # (pad_T+1, C+1,H,W)
        obs = np.concatenate([obs, obs_pad], axis=0) # (H+1, C+1,H,W)

        # Do nothing and get no reward
        actions_pad = np.full((pad_T,), NO_OP_ACTION, dtype=np.int64)
        rewards_pad = np.zeros((pad_T,), dtype=np.float32)
        actions = np.concatenate([actions, actions_pad], axis=0)
        rewards = np.concatenate([rewards, rewards_pad], axis=0)

        infos = list(infos) + [{} for _ in range(pad_T)]
    return TrajectoryWithRew(
        obs=obs,
        acts=actions,
        rews=rewards,
        infos=infos,
        terminal=True
    )

def _append_absorb_indicator(obs: np.ndarray, val: float):
    assert obs.ndim == 3
    C,H,W = obs.shape
    ind = np.full((1,H,W), float(val), dtype=obs.dtype)
    return np.concatenate([obs, ind], axis=0)
