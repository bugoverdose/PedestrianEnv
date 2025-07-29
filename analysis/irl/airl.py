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


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_path)
import pedestrian_env
from pedestrian_env.envs import PedestrianEnv
from analysis.common import CNNFeaturesExtractor
from analysis.irl.reward_net import CustomCNNRewardNet

def run_AIRL(expert_trajs):
    rng = np.random.default_rng(seed=0)
    irl_venv = make_vec_env("PedestrianEnv-v0", n_envs=1, rng=rng, parallel=False) # , env_make_kwargs={"extra_reward_using_crosswalk": True})
    log_dir = "logs/airl/"
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    reward_net = CustomCNNRewardNet(
        observation_space=irl_venv.observation_space,
        action_space=irl_venv.action_space,
    )

    def make_env():
        env = PedestrianEnv()
        env.reset(seed=0)
        return env
    venv = DummyVecEnv([make_env])

    gen_algo = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=128,
        policy_kwargs=dict(
            features_extractor_class=CNNFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        ),
        verbose=1,
    )

    trainer = airl.AIRL(
        demonstrations=expert_trajs,
        demo_batch_size=32,
        venv=venv,
        gen_algo=gen_algo,
        reward_net=reward_net,
        log_dir=log_dir,
        seed=0,
        logger=logger,
    )
    trainer.train(total_timesteps=100_000)

    # torch.save(reward_net.state_dict(), os.path.join(log_dir, "reward_net.pt"))
    torch.save({
        'model_state_dict': reward_net.state_dict(),
        'obs_space': reward_net.observation_space,
        'act_space': reward_net.action_space
    }, os.path.join(log_dir, "reward_net.pt"))

def load_traj(subjId):
    expert_trajectories = []
    data_dir = os.path.join(base_path, "data", f"{subjId}")
    for root, dirs, files in os.walk(data_dir):
        for episode_dir in sorted(dirs):
            print(f"loading {os.path.join(root, episode_dir)}")
            observations = np.load(os.path.join(root, episode_dir, "observations.npy"))
            actions = np.load(os.path.join(root, episode_dir, "actions.npy"))
            rewards = np.load(os.path.join(root, episode_dir, "rewards.npy")).astype(np.float32)
            traj = TrajectoryWithRew(
                obs=observations, # shape: (T+1,), dtype: float32
                acts=actions,     # shape: (T,),  dtype: int64
                rews=rewards,     # shape: (T,),  dtype: float32
                infos=None,
                terminal=True
            )
            expert_trajectories.append(traj)
    return expert_trajectories

# def analyze_reward_function(trainer, irl_venv):
    # reward_fn = trainer.reward_test

    # # reward function을 여러 상태에 대해 평가
    # obs = irl_venv.reset()
    # rew_values = []

    # for _ in range(100):
    #     obs_tensor = torch.tensor(obs, dtype=torch.float32)
    #     rew = reward_fn(obs_tensor, actions=None).detach().cpu().numpy()[0]
    #     rew_values.append(rew)

    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, _, done, _ = irl_venv.step(action)
    #     if done[0]:
    #         obs = irl_venv.reset()

    # # 보상 시계열 시각화
    # plt.plot(rew_values)
    # plt.xlabel("Timestep")
    # plt.ylabel("Predicted Reward")
    # plt.title("AIRL Predicted Reward over Expert Trajectory")
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    run_AIRL(load_traj(100))