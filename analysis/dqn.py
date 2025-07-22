# needed to import `pedestrian_env` module
import sys
import os
import time

import numpy as np
import random
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pedestrian_env.envs import PedestrianEnv
from util import save_plot

def run_DQN(count=1,
            seed=42,
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

    episode_rewards_list = []
    eval_means = []
    training_times = []
    env = Monitor(PedestrianEnv())
    for i in range(count):
        env.reset(seed=seed)
        np.random.seed(seed)
        random.seed(seed)
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/", 
                    seed=seed,
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
        )
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps, tb_log_name="dqn")
        elapsed = time.time() - start_time
        training_times.append(elapsed)

        episode_rewards = env.get_episode_rewards()
        episode_rewards_list.append(episode_rewards)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
        eval_means.append(mean_reward)

    print(f"Average test score: {np.mean(eval_means):.4f} ± {np.std(eval_means):.4f}")
    print(f"Average training time: {np.mean(training_times):.4f} sec ± {np.std(training_times):.4f}s")

    def plot_func():
        # Pad shorter lists to same length (e.g., repeat last value)
        max_len = max(len(r) for r in episode_rewards_list)
        padded_rewards = [r + [r[-1]] * (max_len - len(r)) for r in episode_rewards_list]
        mean_rewards = np.mean(padded_rewards, axis=0)
        std_rewards = np.std(padded_rewards, axis=0)
        plt.plot(mean_rewards)
        plt.fill_between(range(1, len(mean_rewards)+1),
                        mean_rewards - std_rewards,
                        mean_rewards + std_rewards,
                        alpha=0.3)
    save_plot(plot_func, None, "DQN Learning Curve", "Episode", "Reward")

if __name__ == "__main__":
    run_DQN(seed=42)
