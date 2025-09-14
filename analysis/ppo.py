# needed to import `pedestrian_env` module
import sys
import os

import numpy as np
import random

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
    if activation_fn == nn.ReLU:
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
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

def test_policy(model_name, activation_fn, n_eval_episodes=100, seed=42):
    model, env = _load_PPO_model(model_name, activation_fn, render_mode_human=False, seed=seed)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"{model_name}\ntest score: {mean_reward:.4f}")

def visualize_test(model_name, activation_fn, episode_count=20, seed=42):
    model, env = _load_PPO_model(model_name, activation_fn, seed=seed)
    obs = env.reset()
    episode_count = 0
    while episode_count < 10:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"action={action}, reward={reward}, agent=({info[0]['play_infos'][0]}, {info[0]['play_infos'][1]})")
        if done:
            episode_count += 1

def _load_PPO_model(saved_model_name, activation_fn, seed=42, render_mode_human=True):
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
    if activation_fn == nn.ReLU:
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    model = PPO.load(f"saved/ppo/{saved_model_name}", env=env)
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
    test_policy(model_name, activation_fn)

if __name__ == "__main__":
    # model_name="ppo_v6_LeakyReLU_2"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)

    # model_name="ppo_v6_LeakyReLU_1"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v6_LeakyReLU_1
    # test score: 1554.5000
    # test score: 1327.0000

    model_name="ppo_v6_Tanh_1"
    tuning(model_name=model_name,
           net_arch=[256, 256, 256],
           activation_fn=nn.Tanh,
           total_timesteps=1_000_000)
    # ppo_v6_Tanh_1
    # test score: 1522.0000
    # test score: 1553.5000

    # =====================================
    # Best so far
    # model_name="ppo_v5_LeakyReLU_1"
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        activation_fn=nn.LeakyReLU,
    #        total_timesteps=1_000_000)
    # ppo_v5_LeakyReLU_1
    # test score: 1615.0000
    # test score: 1564.0000

    # model_name="ppo_v5_LeakyReLU_2"
    #  net_arch=[512, 256, 256],
    # test score: 1425.5000
    # test score: 1224.5000

    # model_name="ppo_v5_LeakyReLU_3"
    #  net_arch=[512, 256, 128],
    # test score: 1492.5000
    # test score: 1363.5000

    # model_name="ppo_v5_LeakyReLU_4"
    # net_arch=[256, 256, 128],
    # test score: 1538.0000
    # test score: 1425.0000

    # model_name="ppo_v5_LeakyReLU_5"
    # net_arch=[256, 256, 256]
    # total_timesteps=1_500_000
    # ppo_v5_LeakyReLU_5
    # test score: 1519.0000
    # test score: 1586.5000

    # ppo_v3_LeakyReLU_1
    # test score: 1483.0000
    # test score: 1565.0000

    # ppo_v4_LeakyReLU_1
    # test score: 1280.0000
    # test score: 888.5000

    # ppo_v5_SiLU_1
    # test score: 1442.0000
    # test score: 1478.0000

    # ppo_v3_ReLU_1
    # test score: 1500.5000
    # test score: 1636.5000

    # ppo_v5_ReLU_1
    # test score: 1366.0000
    # test score: 1397.5000

    # ppo_v4_ReLU_1
    # test score: 1104.5000
    # test score: 1357.5000

    # ppo_v3_SiLU_1
    # test score: 1570.5000
    # test score: 1247.0000

    # model_name="ppo_v3_SiLU_2" # BAD
    # tuning(model_name=model_name,
    #        net_arch=[256, 256, 256],
    #        activation_fn=nn.SiLU,
    #        learning_rate=3e-4,
    #        total_timesteps=1_000_000)

    # model_name="ppo_v5_Tanh_1"
    # net_arch=[256, 256, 256],
    # test score: 1364.0000
    # test score: 1192.0000

    # ppo_v5_Tanh_2
    # net_arch [512, 256, 256]
    # test score: 1280.5000
    # test score: 1095.5000

    # ppo_v5_Tanh_3
    # net_arch [256, 256, 128]
    # test score: 1292.0000
    # test score: 1224.0000

    # ppo_v3_1
    # test score: 1498.0000
    # test score: 1384.0000

    # model_name="ppo_v3_2"
    # tuning(model_name=model_name,
    #        net_arch=[512, 512, 512],
    #        activation_fn=nn.Tanh,
    #        total_timesteps=1_000_000)
    # net_arch [512, 512, 512]
    # n_steps 512

    # ppo_v3_2
    # test score: 1199.0000
    # test score: 1183.0000