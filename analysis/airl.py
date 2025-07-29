# needed to import `pedestrian_env` module
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import gymnasium
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.types import TrajectoryWithRew
from imitation.data.rollout import make_sample_until, rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.adversarial import gail, airl
from imitation.rewards.reward_nets import BasicRewardNet, CnnRewardNet
from imitation.util.util import make_vec_env
from imitation.util.logger import configure

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
from pedestrian_env.envs import PedestrianEnv

def load_traj(subjId):
    expert_trajectories = []
    data_dir = os.path.join(base_path, "data", f"{subjId}")
    for root, dirs, files in os.walk(data_dir):
        for episode_dir in sorted(dirs):
            observations = np.load(os.path.join(root, episode_dir, "observations.npy"))
            actions = np.load(os.path.join(root, episode_dir, "actions.npy"))
            rewards = np.load(os.path.join(root, episode_dir, "rewards.npy")).astype(np.float32)
            traj = TrajectoryWithRew(
                obs=observations, # shape: (T+1), dtype: float32
                acts=actions,     # shape: (T,),  dtype: int64
                rews=rewards,     # shape: (T,),  dtype: float32
                infos=None,
                terminal=True
            )
            expert_trajectories.append(traj)

if __name__ == "__main__":
    load_traj(1)