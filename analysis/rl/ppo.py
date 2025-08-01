# needed to import `pedestrian_env` module
import sys
import os

import numpy as np
import random

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pedestrian_env.envs import PedestrianEnv
from analysis.common import CNNFeaturesExtractor
from analysis.rl.custom_ppo import CustomPPO

def run_PPO_CnnPolicy(seed=42,
                      features_dim=256, 
                      filters_per_group=3,
                      n_output_channels=[32],
                      kernel_size=3,
                      frame_stack = 0,
                      learning_rate=1e-4,
                      n_steps=2048, # default=2048
                      batch_size=32, # default=64, 
                      n_epochs=2,
                      gamma=0.99,
                      gae_lambda=0.95,
                      clip_range=0.2, # lambda progress: clip_range * progress
                      clip_range_vf=None, # lambda progress: clip_range_vf * progress
                      ent_coef_init=1.0,
                      ent_coef_final=0.01,
                      ent_coef_fraction=0.9,
                      vf_coef=0.5,
                      max_grad_norm=0.5,
                      total_timesteps=500_000,
                      n_eval_episodes=100,
                      extra_reward_using_crosswalk=False,
                      saved_model_name=None,
                      tb_log_name="ppo",
                      verbose=True):

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
    model = CustomPPO("CnnPolicy", env,
                learning_rate=learning_rate,
                n_steps=n_steps, # The number of steps to run for each environment per update
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda, # bias vs variance trade-off for Generalized Advantage Estimator (0 = high bias, low variance; 1 = low bias, high variance)
                clip_range=clip_range, # how much policy can change
                clip_range_vf=clip_range_vf,
                # NOTE: ent_coef = exploit vs explore trade-off (0 = exploit only; 1 = explore only)
                ent_coef_init=ent_coef_init,
                ent_coef_final=ent_coef_final,
                ent_coef_fraction=ent_coef_fraction,
                vf_coef=vf_coef, # policy vs value loss (0=focus on policy , 1=more weight on value loss)
                max_grad_norm=max_grad_norm, # gradient clipping range to prevent gradient from becoming too big during back propagation (0.1 = slow update, 1.0 = more update)
                target_kl=None,
                tensorboard_log="./tb_logs/",
                policy_kwargs=dict(
                    features_extractor_class=CNNFeaturesExtractor,
                    features_extractor_kwargs=dict(
                        features_dim=features_dim,
                        filters_per_group=filters_per_group,
                        n_output_channels=n_output_channels,
                        kernel_size=kernel_size,
                    ),
                ),
                verbose=1 if verbose else 0,
                seed=seed,
    )

    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)

    print("features_dim", features_dim)
    print("filters_per_group", filters_per_group)
    print("n_output_channels", n_output_channels)
    print("kernel_size", kernel_size)
    print("gamma", gamma)
    print("learning_rate", learning_rate)
    print("n_steps", n_steps)
    print("batch_size", batch_size)
    print("n_epochs", n_epochs)
    print("ent_coef_init", ent_coef_init)
    print("ent_coef_final", ent_coef_final)
    print("ent_coef_fraction", ent_coef_fraction)
    print("gae_lambda", gae_lambda)
    print("clip_range", clip_range)
    print("total_timesteps", total_timesteps)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"test score: {mean_reward:.4f}")

    if saved_model_name is not None:
        model.save(f"saved/{saved_model_name}")

def test_policy(model_name, n_eval_episodes=100, seed=42):
    model, env = _load_PPO_model(model_name, render_mode_human=False, seed=seed)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"{model_name}: test score: {mean_reward:.4f}")

def visualize_test(model_name, episode_count=20, seed=42):
    model, env = _load_PPO_model(model_name, seed=seed)
    obs = env.reset()
    episode_count = 0
    while episode_count < 10:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"action={action}, reward={reward}, agent=({info[0]['agent_x']}, {info[0]['agent_y']})")
        if done:
            episode_count += 1

def _load_PPO_model(saved_model_name, render_mode_human=True, frame_stack=0, seed=42):
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
    model = CustomPPO.load(f"saved/{saved_model_name}", env=env)
    return model, env
