import json
import numpy as np
from pathlib import Path

EPISODE_METADATA = "episode_metadata.json"
OBSERVATIONS = "observations.npy"
PLAY_INFOS = "play_infos.npy"
CAR_INFOS = "car_infos.npy"
ACTIONS = "actions.npy"
REWARDS = "rewards.npy"

def data_dir(subjId):
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    parent_dir = current_dir.parent.parent
    data_dir = parent_dir / "data" / str(subjId)
    return data_dir

def get_sorted_episodes(subj_data_dir):
    episodes = []
    for sub_dir in subj_data_dir.iterdir():
        if sub_dir.is_dir():
            episodes.append(sub_dir.name)
    return sorted(episodes)

if __name__ == "__main__":
    subjId = 1
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    for episode in episodes:
        episode_dir = subj_data_dir / episode
        print(episode_dir)

        with open(episode_dir / EPISODE_METADATA, "r") as f:
            metadata = json.load(f)
            print(metadata["road_metadata"][0])
            print(metadata["car_metadata"][0])

        observations = np.load(episode_dir / OBSERVATIONS)
        play_infos   = np.load(episode_dir / PLAY_INFOS, allow_pickle=True)
        car_infos    = np.load(episode_dir / CAR_INFOS, allow_pickle=True)
        actions      = np.load(episode_dir / ACTIONS)
        rewards      = np.load(episode_dir / REWARDS)

        print(observations[0][0][0])
        print(play_infos[0])
        print(car_infos[0])
        print(actions[0])
        print(rewards[0])
