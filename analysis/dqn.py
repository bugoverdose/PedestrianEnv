# needed to import `pedestrian_env` module
import sys
import os

import numpy as np
import random

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pedestrian_env.envs import PedestrianEnv

def run_DQN(seed=42,
            net_arch=[256, 256, 256],
            gamma=0.99, # default=0.99
            learning_rate=1e-4, # default=1e-4
            train_freq=(4, "episode"), # default=(4, "step")
            exploration_fraction= 0.9,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            learning_starts=10_000, # default=50000
            tau=1.0, # default=1 (Hard update only)
            target_update_interval=50, # target network: hard update every 50 steps (default=10000)
            buffer_size=10_000, # Experience Replay: (default=1_000_000)
            batch_size=32, # default=32
            gradient_steps=-1, # auto=-1, default=1
            total_timesteps=500_000,
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
        model.save(f"saved/{saved_model_name}")
        env.save(f"saved/{saved_model_name}.pkl")

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
    model = DQN.load(f"saved/{saved_model_name}", env=env)
    return model, env

def tuning(
        net_arch_depth=3,
        net_arch_width=256,
        gamma=0.99, # default=0.99
        learning_rate=1e-4, # default=1e-4
        train_freq_cnt=4,
        train_freq_type="episode",
        exploration_fraction= 0.9,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        learning_starts=10_000, # default=50000
        target_update_interval=50, # target network: hard update every 50 steps (default=10000)
        buffer_size=10_000, # Experience Replay: (default=1_000_000)
        batch_size=32, # default=32
        total_timesteps=500_000,
        ver = 1,
    ):
    net_arch = [net_arch_width] * net_arch_depth
    model_name = f"dqn_net_arch{net_arch_depth}_{net_arch_width}_g{gamma}_lr{learning_rate}_ts{total_timesteps}_bs{batch_size}_bf{buffer_size}_" \
        + f"exploration_{exploration_fraction}_{exploration_initial_eps}_{exploration_final_eps}_" \
            + f"ls{learning_starts}_train_freq{train_freq_cnt}{train_freq_type}_target_net_update{target_update_interval}_" \
            + f"{ver}_1"
    run_DQN(saved_model_name=model_name, tb_log_name=model_name[:-2],
            net_arch=net_arch,
            gamma=gamma, # default=0.99
            learning_rate=learning_rate, # default=1e-4
            train_freq=(train_freq_cnt, train_freq_type), # default=(4, "step")
            exploration_fraction= exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            learning_starts=learning_starts, # default=50000
            tau=1.0, # default=1 (Hard update only)
            target_update_interval=target_update_interval, # target network: hard update every 50 steps (default=10000)
            buffer_size=buffer_size, # Experience Replay: (default=1_000_000)
            batch_size=batch_size, # default=32
            gradient_steps=-1, # auto=-1, default=1
            total_timesteps=total_timesteps,
            n_eval_episodes=100,
    )
    test_policy(model_name)

if __name__ == "__main__":
    # tuning(net_arch_depth=3, 
    #        net_arch_width=256, 
    #        total_timesteps=1_000_000, 
    #        exploration_fraction=0.5,
    #        exploration_initial_eps=1.0,
    #        exploration_final_eps=0.0,
    #        ver=2)
    model_name = "dqn_net_arch3_256_g0.99_lr0.0001_ts1000000_bs32_bf10000_exploration_0.5_1.0_0.0_ls10000_train_freq4episode_target_net_update50_2_1"
    # net_arch [256, 256, 256]
    # target_update_interval 50
    # buffer_size 10000
    # learning_starts 10000
    # train_freq (4, 'episode')
    # exploration_fraction 0.5
    # exploration_final_eps 0.0
    # learning_rate 0.0001
    # n_eval_episodes 100
    # total_timesteps 1000000
    # test score: 1701.5000
    visualize_test(model_name)
    # test score: 1514.5000

    # tuning(net_arch_depth=3, net_arch_width=256, total_timesteps=1_000_000, ver=2)
    # model_name = "dqn_net_arch3_256_g0.99_lr0.0001_ts1000000_bs32_bf10000_exploration_0.9_1.0_0.01_ls10000_train_freq4episode_target_net_update50_2_1"
    # net_arch [256, 256, 256]
    # target_update_interval 50
    # buffer_size 10000
    # learning_starts 10000
    # train_freq (4, 'episode')
    # exploration_fraction 0.9
    # exploration_final_eps 0.01
    # learning_rate 0.0001
    # n_eval_episodes 100
    # total_timesteps 1000000
    # test score: 1555.0000
    # visualize_test(model_name)
    # test score: 900.0000


    # model_name = "dqn_net_arch3_256_g0.99_lr0.0001_ts500000_bs32_bf10000_exploration_0.9_1.0_0.01_ls10000_train_freq4episode_target_net_update50_2"
    # net_arch [256, 256, 256]
    # target_update_interval 50
    # buffer_size 10000
    # learning_starts 10000
    # train_freq (4, 'episode')
    # exploration_fraction 0.9
    # exploration_final_eps 0.01
    # learning_rate 0.0001
    # n_eval_episodes 100
    # total_timesteps 500000
    # test score: 1515.0000
    # visualize_test(model_name)
    # test score: 1079.5000

    # model_name = "dqn_net_arch3_256_g0.99_lr0.0001_ts500000_bs32_bf10000_exploration_0.9_1.0_0.01_ls10000_train_freq4episode_target_net_update50_1"
    # tuning(net_arch_depth=3, net_arch_width=256) # 1
    # net_arch [256, 256, 256]
    # target_update_interval 50
    # buffer_size 10000
    # learning_starts 10000
    # train_freq (4, 'episode')
    # exploration_fraction 0.9
    # exploration_final_eps 0.01
    # learning_rate 0.0001
    # n_eval_episodes 100
    # total_timesteps 500000
    # test score: 1428.5000
    # visualize_test(model_name)
    # test score: 683.5000

    # tuning(net_arch_depth=3, net_arch_width=512, total_timesteps=1_000_000) # 3
    # model_name = "dqn_net_arch3_512_g0.99_lr0.0001_ts1000000_bs32_bf10000_exploration_0.9_1.0_0.01_ls10000_train_freq4episode_target_net_update50_1"
    # net_arch [512, 512, 512]
    # target_update_interval 50
    # buffer_size 10000
    # learning_starts 10000
    # train_freq (4, 'episode')
    # exploration_fraction 0.9
    # exploration_final_eps 0.01
    # learning_rate 0.0001
    # n_eval_episodes 100
    # total_timesteps 1000000
    # test score: -39.5000
    # visualize_test(model_name)
    # test score: -25.5000
