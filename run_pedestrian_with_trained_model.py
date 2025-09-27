import os
import time
from datetime import datetime

import numpy as np

from pedestrian_env.envs import PedestrianEnv
from analysis.rl.ppo import load_PPO_model, test_policy

def play_episode(model, env, episode_seed):
    obs, info = env.reset(seed=episode_seed)
    episode_metadata = {"env_configuration": info["env_configuration"], "episode_configuration": info["episode_configuration"]}
    play_infos = [info["play_infos"]]
    observations = [obs]
    actions = []
    rewards = []

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if episode_metadata is None:
            episode_metadata = {"env_configuration": info["env_configuration"], "episode_configuration": info["episode_configuration"]}
        play_infos.append(info["play_infos"])
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        if terminated or truncated: break
    return episode_metadata, observations, actions, rewards, play_infos

def play_game(model, base_dir, session_id, max_episodes, base_seed=0):
    episode_count = 0
    env = PedestrianEnv()
    while True:
        if episode_count >= max_episodes: break
        episode_count += 1
        episode_id = session_id * 1000 + episode_count # assumes that each session is less than 1000 episodes
        episode_seed = base_seed + episode_id
        episode_metadata, observations, actions, rewards, play_infos = play_episode(model, env, episode_seed)
        # NOTE: .npy is more lightweight than CSV because it stores data in pure binary format
        os.makedirs(f"{base_dir}/{episode_id:04d}", exist_ok=True)
        np.save(f"{base_dir}/{episode_id:04d}/observations.npy", np.array(observations))
        np.save(f"{base_dir}/{episode_id:04d}/actions.npy", np.array(actions))
        np.save(f"{base_dir}/{episode_id:04d}/rewards.npy", np.array(rewards))
        np.save(f"{base_dir}/{episode_id:04d}/episode_metadata.npy", np.array([episode_metadata], dtype=object), allow_pickle=True)
        np.save(f"{base_dir}/{episode_id:04d}/play_infos.npy", np.array(play_infos, dtype=object), allow_pickle=True)
    env.close()

if __name__ == "__main__":
    optimal_policy_model_name = "ppo_v9_LeakyReLU_norm_obs_F_1"
    test_policy(optimal_policy_model_name)
    model, _ = load_PPO_model(optimal_policy_model_name, render_mode_human=False)

    subj_id = 504
    seed = 0
    max_episodes = 50
    base_dir = f"data/{subj_id}"
    os.makedirs(base_dir, exist_ok=True)
    for session_id in [1, 2]:
        with open(os.path.join(base_dir, f"README_{session_id}.md"), 'w') as f:
            timestamp = time.time()
            dt = datetime.fromtimestamp(timestamp)
            f.write(f"Model: {optimal_policy_model_name}\n" 
                    + f"Subject ID: {subj_id}\n"
                    + f"Session ID: {session_id}\n"
                    + f"Play start time: {dt.year}.{dt.month}.{dt.day} {dt.hour}:{dt.minute}:{dt.second}")
        play_game(model, base_dir=base_dir, session_id=session_id, base_seed=seed, max_episodes=max_episodes)
