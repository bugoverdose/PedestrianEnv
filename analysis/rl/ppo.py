# needed to import `pedestrian_env` module
import sys
import os

import numpy as np
import random
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pedestrian_env.envs import PedestrianEnv

def run_PPO(seed=42,
            net_arch=[256, 256, 256],
            activation_fn=nn.Tanh, # default=nn.Tanh
            learning_rate=1e-4, # default=3e-4
            n_steps=512, # default=2048
            batch_size=32, # default=64
            n_epochs=8, # default=10
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.01, # default=0.0
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            rollout_buffer_class=None,
            rollout_buffer_kwargs=None,
            target_kl=None,
            stats_window_size=100,
            total_timesteps=500_000,
            n_eval_episodes=100,
            saved_model_name=None,
            tb_log_name="PPO",
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
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    model = PPO("MlpPolicy", env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                normalize_advantage=normalize_advantage,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                rollout_buffer_class=rollout_buffer_class,
                rollout_buffer_kwargs=rollout_buffer_kwargs,
                target_kl=target_kl,
                stats_window_size=stats_window_size,
                tensorboard_log="./tb_logs/ppo/",
                policy_kwargs=dict(net_arch=net_arch, activation_fn=activation_fn),
                verbose=1 if verbose else 0,
                seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)

    if verbose:
        learning_rate=learning_rate,
        print("net_arch", net_arch)
        print("n_steps", n_steps)
        print("batch_size", batch_size)
        print("n_epochs", n_epochs)
        print("gamma", gamma)
        print("gae_lambda", gae_lambda)
        print("clip_range", clip_range)
        print("clip_range_vf", clip_range_vf)
        print("normalize_advantage", normalize_advantage)
        print("ent_coef", ent_coef)
        print("vf_coef", vf_coef)
        print("max_grad_norm", max_grad_norm)
        print("use_sde", use_sde)
        print("sde_sample_freq", sde_sample_freq)
        print("rollout_buffer_class", rollout_buffer_class)
        print("rollout_buffer_kwargs", rollout_buffer_kwargs)
        print("target_kl", target_kl)
        print("stats_window_size", stats_window_size)
        print("learning_rate", learning_rate)
        print("n_eval_episodes", n_eval_episodes)
        print("total_timesteps", total_timesteps)

    env.training = False
    env.norm_reward = False
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"test score: {mean_reward:.4f}")
    if saved_model_name is not None:
        model.save(f"saved/ppo/{saved_model_name}")
        env.save(f"saved/ppo/{saved_model_name}.pkl")

def test_policy(model_name, n_eval_episodes=100, seed=42):
    model, env = load_PPO_model(model_name, render_mode_human=False, seed=seed)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"{model_name}\ntest score: {mean_reward:.4f}")

def visualize_test(model_name, episode_count=20, seed=42):
    model, env = load_PPO_model(model_name, seed=seed)
    obs = env.reset()
    episode_count = 0
    while episode_count < 10:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        done, info = dones[0], infos[0]
        [agent_x, agent_y] = info["play_infos"]["agent"]["cur_location"]
        print(f"action={action}, reward={reward}, agent=({agent_x}, {agent_y})")
        if done:
            episode_count += 1

def load_PPO_model(saved_model_name, seed=42, render_mode_human=True):
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
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "saved" / "ppo" / saved_model_name
    model = PPO.load(str(model_path), env=env)
    return model, env

# NOTE: tuning tips
# approx_kl
# - good range: 0.005 ~ 0.03
# - over 0.05 : unstable, policy being updated too much
# - under 0.001 : too weak
# clip_fraction
# - good range: 0.05 ~ 0.3
# - over 0.5 : more than half are getting cut. over-updating
# - near 0 : learning rate too low, update too weak
def tuning(
        model_name,
        net_arch=[256, 256, 256],
        activation_fn=nn.Tanh,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=32,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        total_timesteps=500_000,
    ):
    run_PPO(saved_model_name=model_name,
            tb_log_name=model_name[:-2],
            net_arch=net_arch,
            activation_fn=activation_fn,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            total_timesteps=total_timesteps,
    )
    test_policy(model_name)

# # linear_schedule(2e-4, 1e-4)
# def linear_schedule(start, end):
#     def f(progress):
#         return end + (start - end) * progress # progress: 1->0
#     return f

if __name__ == "__main__":
    # # Takes time to learn, but optimal solution for [height=21, initial_height_padding=0]
    model_name="ppo_v9_LeakyReLU_norm_obs_F_1" # norm_obs=False
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=512,
    #        batch_size=32,
    #        n_epochs=4,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v9_LeakyReLU_norm_obs_F_1
    # test score: 1755.0000
    test_policy(model_name)
    # test score: 1912.0000
    visualize_test(model_name)

    # # almost optimal : goes back even when the car is only on the front lane and can't hit the agent
    # model_name="ppo_v9_Tanh_norm_obs_F_1" # norm_obs=False
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=512,
    #        batch_size=32,
    #        n_epochs=4,
    #        activation_fn=nn.Tanh,
    #        total_timesteps=1_000_000)
    # # ppo_v9_Tanh_norm_obs_F_1
    # # test score: 1817.5000
    # test_policy(model_name)
    # # test score: 1702.5000
    # visualize_test(model_name)

    # model_name="ppo_v9_LeakyReLU_1"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=512,
    #        batch_size=32,
    #        n_epochs=4,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # # ppo_v9_LeakyReLU_1
    # # test score: 1781.5000
    # test_policy(model_name)
    # # test score: 1833.0000
    # visualize_test(model_name)

    # ======================

    # model_name="ppo_v9_ReLU_norm_obs_F_1" # norm_obs=False
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=512,
    #        batch_size=32,
    #        n_epochs=4,
    #        activation_fn=nn.ReLU,
    #        total_timesteps=1_000_000)
    # ppo_v9_ReLU_norm_obs_F_1
    # test score: 1581.5000
    # test score: 1595.0000

    # ======================
    # model_name="ppo_v9_LeakyReLU_4"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=1024,
    #        batch_size=64,
    #        n_epochs=4,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v9_LeakyReLU_4
    # test score: 1741.5000
    # test score: 1679.0000

    # model_name="ppo_v9_LeakyReLU_2"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=512,
    #        batch_size=16,
    #        n_epochs=4,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v9_LeakyReLU_2
    # test score: 1710.5000
    # test score: 1463.5000

    # model_name="ppo_v9_LeakyReLU_3"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=512,
    #        batch_size=64,
    #        n_epochs=4,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v9_LeakyReLU_3
    # est score: 1706.5000
    # test score: 1711.0000

    # model_name="ppo_v9_LeakyReLU_5"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=256,
    #        batch_size=16,
    #        n_epochs=4,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v9_LeakyReLU_5
    # test score: 1561.5000
    # test score: 1380.0000

    # =====================================
    # model_name="ppo_v8_LeakyReLU_7"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=512,
    #        batch_size=32,
    #        n_epochs=4,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v8_LeakyReLU_7
    # test score: 1748.5000
    # test score: 1664.0000

    # model_name="ppo_v8_LeakyReLU_1"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=512,
    #        batch_size=32,
    #        n_epochs=8,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v8_LeakyReLU_1
    # test score: 1747.5000
    # test score: 1616.0000

    # model_name="ppo_v8_LeakyReLU_5"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=1024,
    #        batch_size=64,
    #        n_epochs=10,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v8_LeakyReLU_5
    # test score: 1700.0000
    # test score: 1644.5000

    # model_name="ppo_v8_LeakyReLU_6"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=1024,
    #        batch_size=64,
    #        n_epochs=4,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v8_LeakyReLU_6
    # test score: 1790.0000
    # test score: 1589.5000

    # model_name="ppo_v8_LeakyReLU_2"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=1024,
    #        batch_size=32,
    #        n_epochs=8,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v8_LeakyReLU_2
    # test score: 1638.5000
    # test score: 1511.0000

    # model_name="ppo_v8_LeakyReLU_3"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=512,
    #        batch_size=64,
    #        n_epochs=8,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v8_LeakyReLU_3
    # test score: 1597.0000
    # test score: 1429.0000

    # model_name="ppo_v8_LeakyReLU_4"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=1024,
    #        batch_size=64,
    #        n_epochs=8,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # test_policy(model_name, nn.LeakyReLU)
    # ppo_v8_LeakyReLU_4
    # test score: 1826.5000
    # test score: 1311.5000

    # model_name="ppo_v8_Tanh_1"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        activation_fn=nn.Tanh,
    #        total_timesteps=1_000_000)
    # ppo_v8_Tanh_1
    # test score: 1676.0000
    # test score: 1590.5000

    # model_name="ppo_v8_LeakyReLU_8"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=512,
    #        batch_size=32,
    #        n_epochs=10,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v8_LeakyReLU_8
    # test score: 1644.0000
    # test score: 1630.5000

    # model_name="ppo_v8_LeakyReLU_9"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        n_steps=2048,
    #        batch_size=32,
    #        n_epochs=8,
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v8_LeakyReLU_9
    # test score: 1690.0000
    # test score: 1048.0000
