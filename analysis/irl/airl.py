# needed to import `pedestrian_env` module
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.types import TrajectoryWithRew
from imitation.data.rollout import make_sample_until, rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.adversarial import gail, airl
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from imitation.util.util import make_vec_env
from imitation.util.logger import configure


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# import pedestrian_env
# from pedestrian_env.envs import PedestrianEnv
# from analysis.common import CNNFeaturesExtractor
# from analysis.irl import CustomCNNRewardNet
from analysis.util import data_dir, get_sorted_episodes, load_episode_play_log

def load_traj(subjId):
    expert_trajectories = []
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    for episode in episodes:
        play_log = load_episode_play_log(subj_data_dir, episode)
        observations = play_log["observations"]
        actions = play_log["actions"]
        rewards = play_log["rewards"]
        infos = play_log["play_infos"][1:]
        traj = TrajectoryWithRew(
            # shape: (T+1,), dtype: float32
            obs=observations,
            # shape: (T,), dtype: int64
            acts=actions,
            # shape: (T,), dtype: float32
            rews=rewards,
            # shape: (T,)
            infos=infos,
            terminal=True
        )
        expert_trajectories.append(traj)
    return expert_trajectories

if __name__ == "__main__":
    traj = load_traj(subjId = 1)
