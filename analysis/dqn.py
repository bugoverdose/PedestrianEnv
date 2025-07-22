# needed to import `pedestrian_env` module
import sys
import os
import time

import numpy as np
import random

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pedestrian_env.envs import PedestrianEnv

def run_DQN(seed=42,
            net_arch=[84, 84, 84],
            gamma=0.99, # default=0.99
            learning_rate=5e-4, # default=1e-4
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
            n_eval_episodes=20,
            tb_log_name="dqn",
            verbose=True,
            ):
    print("net_arch", net_arch)
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
        env = PedestrianEnv()
        env.reset(seed=seed)
        return env
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    model = DQN("MlpPolicy", env, seed=seed,
                policy_kwargs=dict(net_arch=net_arch),
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
    env.training = False
    env.norm_reward = False
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"test score: {mean_reward:.4f}")

if __name__ == "__main__":
    run_DQN(seed=42)
