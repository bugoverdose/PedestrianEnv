# needed to import `pedestrian_env` module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gymnasium as gym
import pedestrian_env
from pedestrian_env.envs import PedestrianEnv

import numpy as np

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial import airl
from imitation.util import logger

from analysis.cnn import CNNFeaturesExtractor, CustomCNNRewardNet
from analysis.util import data_dir, get_sorted_episodes, load_episode_play_log
from analysis.irl.fixed_horizon import FixedHorizonAbsorbIndicator, create_fixed_horizon_TrajectoryWithRew

def linear_schedule(start, end):
    def f(progress):
        return end + (start - end) * progress # progress: 1->0
    return f

def run_AIRL(subjId, seed = 42, debugging = False):
    traj, max_step = load_traj(subjId = subjId)

    ppo_n_steps=4_096
    gen_train_timesteps = ppo_n_steps * 8 # 32_768
    total_timesteps = 1_000_000 # about 30 * gen_train_timesteps
    if debugging:
        total_timesteps //= 10

    log_dir = f"./logs/{subjId}/"
    save_dir = f"./saved/{subjId}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    def make_env():
        subj_data_dir = data_dir(subjId)
        episodes = get_sorted_episodes(subj_data_dir)
        episode_seed_range = [int(e.strip()) for e in episodes]
        env = PedestrianEnv(fixed_episode_seed_range=episode_seed_range)
        env = FixedHorizonAbsorbIndicator(env, max_step)
        env.reset(seed=None) # use seed from `fixed_episode_seed_range`
        return env
    venv = DummyVecEnv([make_env])
    _validate_fixed_horizon(venv, traj, max_step)

    gen_algo = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=linear_schedule(1e-4, 3e-5),
        n_steps=ppo_n_steps, # The number of steps to run for each environment per update
        batch_size=1024, # Minibatch size
        n_epochs=10, # Number of epoch when optimizing the surrogate loss
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        clip_range_vf=None,
        ent_coef=0.03,
        vf_coef=0.25,
        max_grad_norm=0.5,
        target_kl=0.02,
        policy_kwargs=dict(
            features_extractor_class=CNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                filters_per_group=5,
                n_output_channels=[64, 64],
                kernel_size=3,
            ),
        ),
        tensorboard_log=f"{log_dir}tb_logs/",
        device="auto",
        verbose=1,
        seed=seed,
    )

    C, _, _ = venv.observation_space.shape
    reward_net = CustomCNNRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        filters_per_group=5,
        n_output_channels=[64, 64],
        kernel_size=3,
        mlp_hidden_size=128,
        # Channel 4: Reward tile
        # Last Channel: Absorbing Indicator
        mask_channels = [4, C-1], 
    )
    reward_net.to(gen_algo.policy.device)

    custom_logger = logger.configure(
        folder=log_dir,
        format_strs=["stdout", "csv", "tensorboard"],
    )

    trainer = airl.AIRL(
        demonstrations=traj,
        venv=venv,
        gen_algo=gen_algo,
        reward_net=reward_net,
        demo_batch_size=512,
        demo_minibatch_size=128, 
        n_disc_updates_per_round=3, # The number of discriminator updates after each round of generator updates in AdversarialTrainer.learn().
        gen_train_timesteps=gen_train_timesteps, # The number of steps to train the generator policy for each iteration.
        log_dir=log_dir,
        custom_logger=custom_logger,
        init_tensorboard=True,
        init_tensorboard_graph=True,
        allow_variable_horizon=False,
    )
    trainer.train(total_timesteps=total_timesteps) 

    torch.save({
        'model_state_dict': reward_net.state_dict(),
        'obs_space': reward_net.observation_space,
        'action_space': reward_net.action_space
    }, f"{save_dir}reward_net.pt")
    gen_algo.save(f"{save_dir}generator_ppo.zip")

def load_traj(subjId):
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    play_logs = [load_episode_play_log(subj_data_dir, episode) for episode in episodes]
    max_step = max(len(play_log["actions"]) for play_log in play_logs)

    expert_trajectories = []
    for play_log in play_logs:
        observations = play_log["observations"]
        actions = play_log["actions"]
        rewards = play_log["rewards"]
        infos = [{"play_infos": pl} for pl in play_log["play_infos"][1:]]
        traj = create_fixed_horizon_TrajectoryWithRew(observations, actions, rewards, infos, max_step)
        expert_trajectories.append(traj)
    return expert_trajectories, max_step

def _validate_fixed_horizon(venv, traj, max_step):
    assert venv.observation_space.shape[0] == traj[0].obs.shape[1]
    T = len(traj[0].acts)
    assert traj[0].obs.shape[0] == T + 1
    env = venv.envs[0]
    obs, info = env.reset()
    ret = 0.0
    for t in range(max_step):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        ret += rew
        if info.get("absorbing", False):
            assert np.allclose(obs[-1], 1.0) and np.allclose(obs[:-1], 0.0)
            assert rew == 0.0
    assert truncated and not terminated

if __name__ == "__main__":
    run_AIRL(subjId = 100)
