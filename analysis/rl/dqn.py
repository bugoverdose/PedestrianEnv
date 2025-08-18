# needed to import `pedestrian_env` module
import sys
import os

import numpy as np
import random

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pedestrian_env.envs import PedestrianEnv
from analysis.cnn import CNNFeaturesExtractor

def run_DQN_CnnPolicy(seed=42,
            features_dim=128,
            filters_per_group=5,
            n_output_channels=[64, 64],
            kernel_size=3,
            frame_stack=0,
            gamma=0.99, # default=0.99
            learning_rate=1e-4, # default=1e-4
            train_freq=(4, "episode"), # default=(4, "step")
            exploration_initial_eps=1.0,
            exploration_fraction= 0.9,
            exploration_final_eps=0.01,
            learning_starts=10_000, # default=50000
            tau=1.0, # default=1 (Hard update only)
            target_update_interval=50, # target network: hard update every 50 steps (default=10000)
            buffer_size=10_000, # Experience Replay: (default=1_000_000)
            batch_size=32, # default=32
            gradient_steps=-1, # auto=-1, default=1
            total_timesteps=500_000,
            n_eval_episodes=100,
            extra_reward_using_crosswalk=False,
            saved_model_name=None,
            tb_log_name="dqn",
            verbose=True,
            ):
    np.random.seed(seed)
    random.seed(seed)
    def make_env():
        env = PedestrianEnv(extra_reward_using_crosswalk=extra_reward_using_crosswalk)
        env.reset(seed=seed)
        return env
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    if frame_stack > 0:
        env = VecFrameStack(env, n_stack=frame_stack)
    model = DQN("CnnPolicy", env, seed=seed,
                policy_kwargs=dict(
                    features_extractor_class=CNNFeaturesExtractor,
                    features_extractor_kwargs=dict(
                        features_dim=features_dim,
                        filters_per_group=filters_per_group,
                        n_output_channels=n_output_channels,
                        kernel_size=kernel_size
                    )
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

    print("features_dim", features_dim)
    print("filters_per_group", filters_per_group)
    print("n_output_channels", n_output_channels)
    print("kernel_size", kernel_size)
    print("target_update_interval", target_update_interval)
    print("buffer_size", buffer_size)
    print("learning_starts", learning_starts)
    print("train_freq", train_freq)
    print("exploration_fraction", exploration_fraction)
    print("exploration_final_eps", exploration_final_eps)
    print("learning_rate", learning_rate)
    print("n_eval_episodes", n_eval_episodes)
    print("total_timesteps", total_timesteps)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"test score: {mean_reward:.4f}")

    if saved_model_name is not None:
        model.save(f"saved/{saved_model_name}")

def test_policy(model_name, n_eval_episodes=100, seed=42):
    model, env = _load_DQN_model(model_name, render_mode_human=False, seed=seed)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"{model_name}: test score: {mean_reward:.4f}")

def visualize_test(model_name, episode_count=20, seed=42):
    model, env = _load_DQN_model(model_name, seed=seed)
    obs = env.reset()
    episode_count = 0
    while episode_count < 10:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"action={action}, reward={reward}, agent=({info[0]['agent_x']}, {info[0]['agent_y']})")
        if done:
            episode_count += 1

def _load_DQN_model(saved_model_name, seed=42, frame_stack=0, render_mode_human=True):
    np.random.seed(seed)
    random.seed(seed)

    def make_env():
        if render_mode_human:
            env = PedestrianEnv(render_mode = "human", realtime=True, gameover_screen_time=2000, render_sprite=True)
        else:
            env = PedestrianEnv()
        env.reset(seed=seed)
        return env

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    if frame_stack > 0:
        env = VecFrameStack(env, n_stack=frame_stack)
    model = DQN.load(f"saved/{saved_model_name}", env=env)
    return model, env

if __name__ == "__main__":
    model_name = "dqn_5_64_64_fd128_kernel3_1"
    run_DQN_CnnPolicy(saved_model_name=model_name, tb_log_name=model_name[:-2])
    test_policy(model_name)
    # test score: 1652.5000
    visualize_test(model_name)
    # check 6_best_dqn.mov
