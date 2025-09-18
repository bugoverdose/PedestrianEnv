# needed to import `pedestrian_env` module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import random
import numpy as np

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial import airl
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import logger

from pedestrian_env.envs import PedestrianEnv
from util import load_subject_play_log, load_traj, FixedHorizonEnvWrapper

def train_AIRL(
        subjId,
        ppo_n_steps,
        gen_train_timesteps,
        airl_train_n_rounds,
        gen_replay_buffer_capacity=None,
        n_disc_updates_per_round=2,
        disc_optimizer_lr=1e-3,
        disc_optimizer_weight_decay=0,
        reward_net_hid_sizes=[32, 32],
        reward_net_activation=nn.Tanh,
        use_fixed_episodes=True,
        seed=42,
        sub_dir=None,
    ):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log_dir = f"./tb_logs/{subjId}/{sub_dir}"
    save_dir = f"./saved/{subjId}/{sub_dir}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    def make_env():
        _, max_traj_size, episodes = load_subject_play_log(subjId = subjId)
        if use_fixed_episodes:
            env = PedestrianEnv(fixed_episode_seed_range=episodes)
            env = FixedHorizonEnvWrapper(env, max_traj_size)
            env.reset(seed=None) # use seed from `fixed_episode_seed_range`
        else:
            env = PedestrianEnv()
            env = FixedHorizonEnvWrapper(env, max_traj_size)
            env.reset(seed=seed)
        return env
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True) # `norm_obs=False` finds the optimal policy for PPO

    gen_algo = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=ppo_n_steps,
        batch_size=32,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        tensorboard_log=f"{log_dir}ppo/",
        policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn=nn.Tanh), # or nn.LeakyReLU
        verbose=1,
        seed=seed,
    )
    reward_net = BasicRewardNet(
        env.observation_space,
        env.action_space,
        hid_sizes=reward_net_hid_sizes, # (128, 128)
        activation=reward_net_activation, # nn.Tanh
    )
    trainer = airl.AIRL(
        demonstrations=load_traj(subjId = subjId),
        demo_batch_size=256,
        # demo_minibatch_size=demo_batch_size,
        venv=env,
        gen_algo=gen_algo,
        reward_net=reward_net,

        # The number of discriminator updates after each round of generator updates in AdversarialTrainer.learn().
        # # default: 2
        n_disc_updates_per_round=n_disc_updates_per_round,

        # The number of steps to train the generator policy for each iteration.
        # If None, then defaults to the batch size (for on-policy) or number of environments (for off-policy).
        # default: gen_algo.n_steps * gen_algo_env.num_envs
        gen_train_timesteps=gen_train_timesteps,

        # The capacity of the generator replay buffer (the number of obs-action-obs samples from the generator that can be stored). 
        # By default this is equal to `gen_train_timesteps`, meaning that we sample only from the most recent batch of generator samples.
        # default: gen_train_timesteps
        gen_replay_buffer_capacity=gen_train_timesteps * gen_replay_buffer_capacity,

        # default for Adam: dict(lr=1e-3, weight_decay=0),
        disc_opt_kwargs=dict(lr=disc_optimizer_lr, weight_decay=disc_optimizer_weight_decay),

        log_dir=log_dir,
        custom_logger=logger.configure(
            folder=log_dir,
            format_strs=["stdout", "csv", "tensorboard"],
        ),
        init_tensorboard=True, # If True, makes various discriminator TensorBoard summaries.
        init_tensorboard_graph=True, # If both this and `init_tensorboard` are True, then write a Tensorboard graph summary to disk.
    )

    # train: `train_gen(self.gen_train_timesteps)` => `train_disc` => `callback(round)`
    eval_env = make_env()
    eval_log = logger.configure(
        folder=f"{log_dir}/eval",
        format_strs=["stdout", "csv", "tensorboard"],
    )
    def evaluate_callback(round):
        mean_rew, std_rew = evaluate_policy(gen_algo, eval_env, n_eval_episodes=100, deterministic=True)
        eval_log.record("eval/mean_reward", float(mean_rew))
        eval_log.record("eval/std_reward", float(std_rew))
        eval_log.dump(step=int(gen_algo.num_timesteps))
    trainer.train(total_timesteps=gen_train_timesteps * airl_train_n_rounds, callback=evaluate_callback)
    eval_env.close()

    torch.save({
        'model_state_dict': reward_net.state_dict(),
        'obs_space': reward_net.observation_space,
        'action_space': reward_net.action_space
    }, f"{save_dir}reward_net.pt")
    gen_algo.save(f"{save_dir}generator_ppo.zip")

if __name__ == "__main__":
    ppo_n_steps=512
    # gen_train_timesteps = ppo_n_steps * 1
    # airl_train_n_rounds = 3000 # about (1_500_000 // gen_train_timesteps)
    # total_timesteps = gen_train_timesteps * airl_train_n_rounds
    # print(f"gen_train_timesteps={gen_train_timesteps}, airl_train_n_rounds={airl_train_n_rounds}, total_timesteps={total_timesteps}")

    # Best so far
    # raw/disc/disc_acc_expert = 0.5 정도
    # raw/disc/disc_acc_gen = 0.6 => 0.7로 우상향 => 해결 필요
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     gen_replay_buffer_capacity=3,
    #     airl_train_n_rounds=4000,
    #     n_disc_updates_per_round=1,
    #     disc_optimizer_lr=1e-3,
    #     disc_optimizer_weight_decay=1e-4,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_256_3_1_disc_lr1e-3_disc_wd1e-4/",
    # )

    train_AIRL(
        subjId=500,
        ppo_n_steps=ppo_n_steps,
        gen_train_timesteps=ppo_n_steps * 1,
        gen_replay_buffer_capacity=3,
        airl_train_n_rounds=4000,
        n_disc_updates_per_round=1,
        disc_optimizer_lr=1e-3,
        disc_optimizer_weight_decay=1e-4,
        reward_net_hid_sizes=[256, 256, 256],
        reward_net_activation=nn.Tanh,
        use_fixed_episodes=False,
        sub_dir="3_256_3_1_disc_lr1e-3_disc_wd1e-4_F/",
    )

    # ===========================================

    # mean_reward = 1500 => 805.8081
    # mean/disc/disc_acc = 0.6756
    # mean/disc/disc_acc_expert = 0.6142
    # mean/disc/disc_acc_gen = 0.737
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     gen_replay_buffer_capacity=3,
    #     airl_train_n_rounds=4000,
    #     n_disc_updates_per_round=1,
    #     disc_optimizer_lr=1e-3,
    #     disc_optimizer_weight_decay=0,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_256_3_1/",
    # )

    # similar to 3_256_3_1
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     gen_replay_buffer_capacity=3,
    #     airl_train_n_rounds=4000,
    #     n_disc_updates_per_round=1,
    #     disc_optimizer_lr=1e-3,
    #     disc_optimizer_weight_decay=0,
    #     reward_net_hid_sizes=[256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="2_256_3_1/",
    # )

    # ===========================================

    # takes almost 4000 rounds for the accuracy to drop near 0.5
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     gen_replay_buffer_capacity=3,
    #     airl_train_n_rounds=4000,
    #     n_disc_updates_per_round=1,
    #     disc_optimizer_lr=1e-4,
    #     disc_optimizer_weight_decay=0,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_256_3_1_disc_lr1e-4_disc_wd0/",
    # )

    # didn't drop to 0.5 during 4000 steps
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     gen_replay_buffer_capacity=3,
    #     airl_train_n_rounds=4000,
    #     n_disc_updates_per_round=1,
    #     disc_optimizer_lr=1e-4,
    #     disc_optimizer_weight_decay=1e-4,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_256_3_1_disc_lr1e-4_disc_wd1e-4/",
    # )

    # BAD: hid_sizes=[128, 128]
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     gen_replay_buffer_capacity=3,
    #     airl_train_n_rounds=4000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[128, 128],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="2_128_3_1/",
    # )

    # worse than 3_256_3_1
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     gen_replay_buffer_capacity=10,
    #     airl_train_n_rounds=4000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_256_10_1/",
    # )

    # episode reward < 1000, accuracy = 0.8
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 2,
    #     gen_replay_buffer_capacity=3,
    #     airl_train_n_rounds=4000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_256_3_2/",
    # )

    # episode reward < 1000, accuracy = 0.8
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 4,
    #     gen_replay_buffer_capacity=3,
    #     airl_train_n_rounds=4000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_256_3_4/",
    # )

    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     gen_replay_buffer_capacity=3,
    #     airl_train_n_rounds=12000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_256_3/",
    # )

    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     gen_replay_buffer_capacity=1,
    #     airl_train_n_rounds=4000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_256_1/",
    # )

    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     gen_replay_buffer_capacity=2,
    #     airl_train_n_rounds=8000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_256_2/",
    # )


    # airl_train_n_rounds = 3000
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     airl_train_n_rounds=3000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[128, 128, 128],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_128_2/",
    # )

    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     airl_train_n_rounds=3000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[256, 256, 256],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_128_4/",
    # )

    # Terrible
    # 500/3_128_3
    # airl_train_n_rounds = 3000
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     airl_train_n_rounds=3000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[128, 128, 128],
    #     reward_net_activation=nn.LeakyReLU,
    #     sub_dir="3_128_3/",
    # )

    # # airl_train_n_rounds = 3000
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     airl_train_n_rounds=3000,
    #     n_disc_updates_per_round=2,
    #     reward_net_hid_sizes=[128, 128, 128],
    #     reward_net_activation=nn.Tanh,
    #     sub_dir="3_128_1/",
    # )

    # airl_train_n_rounds = 2000
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     airl_train_n_rounds=2000,
    #     n_disc_updates_per_round=2,
    #     reward_net_hid_sizes=[32, 32],
    #     reward_net_activation=nn.ReLU,
    #     sub_dir="1/",
    # )
    
    # airl_train_n_rounds = 3000
    # train_AIRL(
    #     subjId=500,
    #     ppo_n_steps=ppo_n_steps,
    #     gen_train_timesteps=ppo_n_steps * 1,
    #     airl_train_n_rounds=3000,
    #     n_disc_updates_per_round=1,
    #     reward_net_hid_sizes=[32, 32],
    #     reward_net_activation=nn.ReLU,
    #     sub_dir="2/",
    # )
    
    # load_traj(subjId=1)
