# needed to import `pedestrian_env` module
import sys
import os

import numpy as np
import random

import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pedestrian_env.envs import PedestrianEnv

def run_DQN(seed=42,
            net_arch=[256, 256, 256],
            activation_fn=nn.Tanh, # default=nn.Tanh
            learning_rate=1e-4, # default=1e-4
            buffer_size=10_000, # Experience Replay: (default=1_000_000)
            learning_starts=10_000, # default=50000
            batch_size=32, # default=32
            tau=1.0, # default=1 (Hard update only)
            gamma=0.99, # default=0.99
            train_freq=(4, "episode"), # default=(4, "step")
            gradient_steps=-1, # auto=-1, default=1
            target_update_interval=50, # target network: hard update every 50 steps (default=10000)
            exploration_fraction= 0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.0,
            total_timesteps=1_000_000,
            n_eval_episodes=100,
            saved_model_name=None,
            tb_log_name="dqn",
            verbose=True,
    ):
    np.random.seed(seed)
    random.seed(seed)
    def make_env():
        env = PedestrianEnv()
        env.reset(seed=seed)
        return env
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    model = DQN("MlpPolicy", env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts, 
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                target_update_interval=target_update_interval,
                exploration_final_eps=exploration_final_eps,
                exploration_initial_eps=exploration_initial_eps,
                exploration_fraction=exploration_fraction,
                tensorboard_log="./tb_logs/dqn/",
                policy_kwargs=dict(net_arch=net_arch, activation_fn=activation_fn),
                verbose=1 if verbose else 0,
                seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)

    if verbose:
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

    env.training = False
    env.norm_reward = False
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"test score: {mean_reward:.4f}")
    if saved_model_name is not None:
        model.save(f"saved/dqn/{saved_model_name}")
        env.save(f"saved/dqn/{saved_model_name}.pkl")

def test_policy(model_name, n_eval_episodes=100, seed=42):
    model, env = _load_DQN_model(model_name, render_mode_human=False, seed=seed)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"{model_name}\ntest score: {mean_reward:.4f}")

def visualize_test(model_name, episode_count=20, seed=42):
    model, env = _load_DQN_model(model_name, seed=seed)
    obs = env.reset()
    episode_count = 0
    while episode_count < 10:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"action={action}, reward={reward}, agent=({info[0]['play_infos'][0]}, {info[0]['play_infos'][1]})")
        if done:
            episode_count += 1

def _load_DQN_model(saved_model_name, seed=42, render_mode_human=True):
    np.random.seed(seed)
    random.seed(seed)

    def make_env():
        if render_mode_human:
            env = PedestrianEnv(render_mode = "human", realtime=True, gameover_screen_time=2000)
        else:
            env = PedestrianEnv()
        env.reset(seed=seed)
        return env

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    model = DQN.load(f"saved/dqn/{saved_model_name}", env=env)
    return model, env

def tuning(
        model_name,
        net_arch=[256, 256, 256],
        activation_fn=nn.Tanh,
        gamma=0.99,
        learning_rate=1e-4,
        train_freq_cnt=4,
        train_freq_type="episode",
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.0,
        learning_starts=10_000,
        target_update_interval=50,
        buffer_size=10_000,
        batch_size=32,
        total_timesteps=1_000_000,
    ):
    run_DQN(saved_model_name=model_name,
            tb_log_name=model_name[:-2],
            net_arch=net_arch,
            activation_fn=activation_fn,
            gamma=gamma,
            learning_rate=learning_rate,
            train_freq=(train_freq_cnt, train_freq_type),
            exploration_fraction= exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            learning_starts=learning_starts,
            tau=1.0,
            target_update_interval=target_update_interval,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gradient_steps=-1,
            total_timesteps=total_timesteps,
            n_eval_episodes=100,
    )
    test_policy(model_name)

# Tanh > LeakyReLU in DQN
if __name__ == "__main__":
    model_name="dqn_v7_Tanh_1"
    tuning(model_name=model_name,
           net_arch=[256, 256, 256],
           activation_fn=nn.Tanh,
           total_timesteps=1_000_000,
           exploration_fraction=0.5,
           exploration_initial_eps=1.0,
           exploration_final_eps=0.0)
    # dqn_v7_Tanh_1
    # test score: 1580.5000
    # test score: 1296.0000

    # dqn_v5_Tanh_2
    # test score: 1770.5000
    # test score: 1586.5000

    model_name="dqn_v7_LeakyReLU_1"
    tuning(model_name=model_name,
           net_arch=[256, 256, 256],
           activation_fn=nn.LeakyReLU,
           total_timesteps=1_000_000,
           exploration_fraction=0.5,
           exploration_initial_eps=1.0,
           exploration_final_eps=0.0)
    # dqn_v7_LeakyReLU_1
    # test score: 1754.0000
    # test score: 1655.0000

    # dqn_v5_LeakyReLU_2
    # test score: 1666.5000
    # test score: 1161.5000

    # ======================
    # Best so far

    # model_name="dqn_v5_Tanh_1"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        activation_fn=nn.Tanh,
    #        total_timesteps=1_500_000,
    #        exploration_fraction=0.5,
    #        exploration_initial_eps=1.0,
    #        exploration_final_eps=0.0)
    # dqn_v5_Tanh_1
    # test score: 1561.0000
    # test score: 1512.5000


    # model_name="dqn_v5_LeakyReLU_1"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_500_000,
    #        exploration_fraction=0.5,
    #        exploration_initial_eps=1.0,
    #        exploration_final_eps=0.0)
    # dqn_v5_LeakyReLU_1
    # test score: 1524.0000
    # test score: 1549.0000

    # model_name="dqn_v6_Tanh_1"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        activation_fn=nn.Tanh,
    #        total_timesteps=1_000_000,
    #        exploration_fraction=0.5,
    #        exploration_initial_eps=1.0,
    #        exploration_final_eps=0.0)
    # dqn_v6_Tanh_1
    # test score: 1705.0000
    # test score: 1706.0000

    # model_name="dqn_v6_LeakyReLU_1"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000,
    #        exploration_fraction=0.5,
    #        exploration_initial_eps=1.0,
    #        exploration_final_eps=0.0)
    # dqn_v6_LeakyReLU_1
    # test score: 1776.5000
    # test score: 1347.5000