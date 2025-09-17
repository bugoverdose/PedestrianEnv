import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import gymnasium as gym

from imitation.data.types import TrajectoryWithRew

from analysis.util import data_dir, get_sorted_episodes, load_episode_full_log

# TODO: data augmentation 

def load_subject_play_log(subjId):
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    full_log_list = [load_episode_full_log(subj_data_dir, episode) for episode in episodes]
    max_traj_size = max(len(full_log["actions"]) for full_log in full_log_list)
    episodes = [int(e.strip()) for e in episodes]
    return full_log_list, max_traj_size, episodes

def load_traj(subjId):
    full_log_list, max_traj_size, _ = load_subject_play_log(subjId)
    expert_trajectories = []
    for full_log in full_log_list:
        episode_metadata = full_log["episode_metadata"]
        observations = full_log["observations"]
        actions = full_log["actions"]
        rewards = full_log["rewards"]
        infos = [{"play_infos": pl, "episode_metadata": episode_metadata} for pl in full_log["play_infos"][1:]]
        traj = create_fixed_horizon_trajectory(observations, actions, rewards, infos, max_traj_size)
        expert_trajectories.append(traj)
    return expert_trajectories

class FixedHorizonEnvWrapper(gym.Wrapper):
    """
    Add absorbing indicator feature for all the observations (0=ongoing, 1=truncated)

    Add additional transitions until reaching `max_traj_size`.
    `truncated = True` only when reaching `max_traj_size`

    (check `Variable Horizon Environments Considered Harmful`_).

    .. _Variable Horizon Environments Considered Harmful: https://imitation.readthedocs.io/en/latest/main-concepts/variable_horizon.html
    """
    def __init__(self, env, max_traj_size: int):
        super().__init__(env)
        self.max_traj_size = int(max_traj_size)
        self.cur_step = 0
        self.is_absorbing = False

        assert isinstance(env.observation_space, gym.spaces.Box)
        low  = env.observation_space.low
        high = env.observation_space.high
        obs_space_size = len(low) + 1
        assert obs_space_size == 114
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, [0.0]], axis=0),
            high=np.concatenate([high, [1.0]], axis=0),
            dtype=env.observation_space.dtype
        )
        self._absorbing_obs = _make_absorbing_obs(obs_space_size)

    def reset(self, **kw):
        self.cur_step = 0
        self.is_absorbing = False
        obs, info = self.env.reset(**kw)
        return _wrap_non_absorbing_obs(obs), info

    def step(self, action):
        self.cur_step += 1
        if not self.is_absorbing:
            obs, rew, terminated, truncated, info = self.env.step(action)
            obs = _wrap_non_absorbing_obs(obs)
            if terminated or truncated:
                self.is_absorbing = True # start absorbing after current transition
        else:
            obs = self._absorbing_obs
            rew = 0.0
            info = {}
        truncated = self.cur_step >= self.max_traj_size
        return obs, rew, False, truncated, info

def create_fixed_horizon_trajectory(
        observations, # shape: (T+1,), dtype: float32
        actions,      # shape: (T,), dtype: int64
        rewards,      # shape: (T,), dtype: float32
        infos,        # shape: (T,)
        max_traj_size
    ):
    T = len(actions)
    assert len(observations) == T+1
    assert len(rewards) == T
    assert len(infos) == T
    assert max_traj_size >= T

    # add absorbing indicator feature (0=ongoing, 1=truncated)
    observations = [_wrap_non_absorbing_obs(obs) for obs in observations]
    obs_space_size = len(observations[0])
    assert obs_space_size == 114
    pad_T = max_traj_size - T
    if pad_T > 0:
        absorbing_obs = _make_absorbing_obs(obs_space_size)
        observations = observations + [absorbing_obs]*pad_T # shape: (max_traj_size+1, obs_space_size)

        # do nothing and get no reward
        actions_pad = np.full((pad_T), 0, dtype=np.int64) # NOTE: 0 == Action.NOTHING.value
        rewards_pad = np.zeros((pad_T), dtype=np.float32)
        actions = np.concatenate([actions, actions_pad], axis=0)
        rewards = np.concatenate([rewards, rewards_pad], axis=0)
        
        # fill up empty infos 
        infos = list(infos) + [{} for _ in range(pad_T)]

    return TrajectoryWithRew(
        obs=np.asarray(observations),
        acts=actions,
        rews=rewards,
        infos=infos,
        terminal=True
    )

def _make_absorbing_obs(obs_space_size):
    absorbing_obs = np.zeros(obs_space_size, dtype=np.float32)
    absorbing_obs[-1] = 1.0
    return absorbing_obs

def _wrap_non_absorbing_obs(obs):
    return np.concatenate([obs, [0]], axis=0)
