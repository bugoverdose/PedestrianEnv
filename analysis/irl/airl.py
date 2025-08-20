# needed to import `pedestrian_env` module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gymnasium as gym
import pedestrian_env
from pedestrian_env.envs import PedestrianEnv

import random
import numpy as np

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial import airl
from imitation.util import logger

from analysis.cnn import CNNFeaturesExtractor, CustomCNNRewardNet
from analysis.util import data_dir, get_sorted_episodes, load_episode_play_log
from analysis.irl.fixed_horizon import FixedHorizonAbsorbIndicator, create_fixed_horizon_TrajectoryWithRew

def run_AIRL(subjId, seed = 42, debugging = False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    traj, max_step = load_traj(subjId = subjId)

    ppo_n_steps=512
    gen_train_timesteps = ppo_n_steps * 32
    gen_replay_buffer_capacity = gen_train_timesteps * 2
    total_timesteps = gen_train_timesteps * 50
    if debugging:
        total_timesteps //= 10

    log_dir = f"./logs/{subjId}/"
    save_dir = f"./saved/{subjId}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    episode_seed_range = [int(e.strip()) for e in episodes]

    venv = SubprocVecEnv([make_env_fn(max_step, episode_seed_range, i) for i in range(4)])
    _validate_fixed_horizon(episode_seed_range, max_step)

    gen_algo = PPO(
        policy="CnnPolicy",
        policy_kwargs=dict(
            features_extractor_class=CNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                filters_per_group=5,
                n_output_channels=[64, 64],
                kernel_size=3,
            ),
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs=dict(weight_decay=1e-4),
        ),
        env=venv,
        learning_rate=linear_schedule(2e-4, 1e-4),
        n_steps=ppo_n_steps, # The number of steps to run for each environment per update
        batch_size=512, # Minibatch size
        n_epochs=5, # Number of epoch when optimizing the surrogate loss
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=0.5,
        target_kl=None, # 0.02,
        tensorboard_log=f"{log_dir}tb_logs/",
        device="auto",
        verbose=1,
        seed=seed,
    )
    # gen_algo.policy.optimizer = adamw_with_decay(gen_algo.policy, lr=gen_algo.learning_rate(1), wd=1e-4)

    C, _, _ = venv.observation_space.shape
    reward_net = CustomCNNRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        filters_per_group=5,
        n_output_channels=[64, 64],
        kernel_size=3,
        mlp_hidden_size=256,
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
        demo_batch_size=1024,
        demo_minibatch_size=256, 
        n_disc_updates_per_round=2, # The number of discriminator updates after each round of generator updates in AdversarialTrainer.learn().
        gen_train_timesteps=gen_train_timesteps, # The number of steps to train the generator policy for each iteration.
        gen_replay_buffer_capacity=gen_replay_buffer_capacity,
        disc_opt_cls=torch.optim.AdamW,
        disc_opt_kwargs={"lr": 1e-4, "weight_decay": 1e-4},
        log_dir=log_dir,
        custom_logger=custom_logger,
        init_tensorboard=True,
        init_tensorboard_graph=True,
        allow_variable_horizon=False,
    )

    eval_env = make_env_fn(max_step, episode_seed_range)()
    trainer.train(
        total_timesteps=total_timesteps,
        callback=add_eval(gen_algo, eval_env, custom_logger, n_episodes=10)
    )
    eval_env.close()

    torch.save({
        'model_state_dict': reward_net.state_dict(),
        'obs_space': reward_net.observation_space,
        'action_space': reward_net.action_space
    }, f"{save_dir}reward_net.pt")
    gen_algo.save(f"{save_dir}generator_ppo.zip")

class CastRewardToFloat(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.float32(reward)) 

def make_env_fn(max_step, episode_seed_range, seed_start_idx = 0):
    if seed_start_idx > 0:
        episode_seed_range = episode_seed_range[seed_start_idx:] + episode_seed_range[:seed_start_idx]
    def _f():
        env = PedestrianEnv(fixed_episode_seed_range=episode_seed_range)
        env = FixedHorizonAbsorbIndicator(env, max_step)
        env = CastRewardToFloat(env)
        env.reset(seed=None) # use seed from `fixed_episode_seed_range`
        return env
    return _f

def linear_schedule(start, end):
    def f(progress):
        return end + (start - end) * progress # progress: 1->0
    return f

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

def add_eval(gen_algo, eval_env, log, n_episodes=10):
    def _cb(round_idx: int):
        mean_rew, std_rew = evaluate_policy(
            gen_algo,
            eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True
        )
        # log performance metric
        log.record("eval/mean_reward", float(mean_rew))
        log.record("eval/std_reward", float(std_rew))
        log.dump(step=int(gen_algo.num_timesteps))
    return _cb

def _validate_fixed_horizon(episode_seed_range, max_step):
    env = make_env_fn(max_step, episode_seed_range)()
    try:
        obs, info = env.reset()
        ret = 0.0
        terminated = truncated = False
        for t in range(max_step):
            a = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(a)
            ret += rew
            if info.get("absorbing", False):
                assert np.allclose(obs[-1], 1.0) and np.allclose(obs[:-1], 0.0)
                assert rew == 0.0
            if terminated or truncated:
                break
        assert truncated and not terminated
    finally:
        env.close()

# no decay for bias/Norm
def adamw_with_decay(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if n.endswith("bias") or "norm" in n.lower() or "bn" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )

if __name__ == "__main__":
    run_AIRL(subjId = 100)
