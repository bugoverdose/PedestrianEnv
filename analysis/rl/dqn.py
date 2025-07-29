# needed to import `pedestrian_env` module
import sys
import os

import numpy as np
import random

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pedestrian_env.envs import PedestrianEnv

class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512,
                 kernel_size = 3):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0] # number of channels
        # 5 filters per each channel => 5 feature map for each of the input channel
        n_output_channels = n_input_channels * 5
        # NOTE: appropriate kernel_size, padding combinations if stride=1
        # kernel_size = 3, padding = 1
        # kernel_size = 5, padding = 2
        # kernel_size = 7, padding = 3
        padding = (kernel_size - 1) // 2
        self.cnn = nn.Sequential(
            # Group Conv: convolution for each channel. 
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=n_input_channels),
            nn.ReLU(),
            # use multiple feature map together
            nn.Conv2d(n_output_channels, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            # CNN to vector
            nn.Flatten()
        )

        # Compute output shape
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def run_DQN_CnnPolicy(seed=42,
            features_dim=512, kernel_size=3,
            gamma=0.99, # default=0.99
            learning_rate=1e-4, # default=1e-4
            train_freq = 1, # default=4
            exploration_initial_eps=1.0,
            exploration_fraction = 0.6,
            exploration_final_eps = 0.05,
            learning_starts = 2_000, # default=50000
            tau=1.0, # default=1 (Hard update only)
            target_update_interval = 500, # target network: hard update every `target_update_interval` steps (default=10000)
            buffer_size = 10_000, # Experience Replay: (default=1_000_000)
            batch_size=32, # default=32
            gradient_steps=1, # default=1
            total_timesteps=150_000,
            n_eval_episodes=100,
            extra_reward_using_crosswalk=False,
            saved_model_name=None,
            tb_log_name="dqn",
            verbose=True,
            ):
    print("target_update_interval", target_update_interval)
    print("buffer_size", buffer_size)
    print("learning_starts", learning_starts)
    print("train_freq", train_freq)
    print("exploration_fraction", exploration_fraction)
    print("exploration_final_eps", exploration_final_eps)
    print("learning_rate", learning_rate)
    print("n_eval_episodes", n_eval_episodes)
    print("total_timesteps", total_timesteps)

    np.random.seed(seed)
    random.seed(seed)
    def make_env():
        env = PedestrianEnv(extra_reward_using_crosswalk=extra_reward_using_crosswalk)
        env.reset(seed=seed)
        return env
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    model = DQN("CnnPolicy", env, seed=seed,
                policy_kwargs=dict(
                    features_extractor_class=SimpleCNN,
                    features_extractor_kwargs=dict(features_dim=features_dim, kernel_size=kernel_size)
                ),
                gamma=gamma,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts, 
                batch_size=batch_size,
                tau=tau,
                target_update_interval=target_update_interval,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                exploration_fraction=exploration_fraction,
                verbose=1 if verbose else 0,
                tensorboard_log="./tb_logs/",
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"test score: {mean_reward:.4f}")

    if saved_model_name is not None:
        model.save(f"saved/{saved_model_name}")

def visualize_test(model_name, episode_count=20, seed=42):
    model, env = _load_DQN_model(model_name, seed)
    obs = env.reset()
    episode_count = 0
    while episode_count < 10:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"action={action}, reward={reward}, agent=({info[0]['agent_x']}, {info[0]['agent_y']})")
        if done:
            episode_count += 1

def _load_DQN_model(saved_model_name, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    def make_env():
        env = PedestrianEnv(render_mode = "human", realtime=True, gameover_screen_time=2000, render_sprite=True)
        env.reset(seed=seed)
        return env

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    model = DQN.load(f"saved/{saved_model_name}", env=env)
    return model, env

if __name__ == "__main__":
    pass
