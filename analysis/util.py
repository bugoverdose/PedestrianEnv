import json
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

EPISODE_METADATA = "episode_metadata.json"
OBSERVATIONS = "observations.npy"
PLAY_INFOS = "play_infos.npy"
CAR_INFOS = "car_infos.npy"
ACTIONS = "actions.npy"
REWARDS = "rewards.npy"

def data_dir(subjId):
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    data_dir = current_dir.parent / "data" / str(subjId)
    return data_dir

def get_sorted_episodes(subj_data_dir):
    episodes = []
    for sub_dir in subj_data_dir.iterdir():
        if sub_dir.is_dir():
            episodes.append(sub_dir.name)
    return sorted(episodes)

def load_episode_play_log(subj_data_dir, episode):
    episode_dir = subj_data_dir / episode
    with open(episode_dir / EPISODE_METADATA, "r") as f:
        metadata = json.load(f)
    return {
        "road_metadata": metadata["road_metadata"],
        "car_metadata": metadata["car_metadata"],
        "observations": np.load(episode_dir / OBSERVATIONS),
        "play_infos": np.load(episode_dir / PLAY_INFOS, allow_pickle=True),
        "car_infos": np.load(episode_dir / CAR_INFOS, allow_pickle=True),
        "actions": np.load(episode_dir / ACTIONS),
        "rewards": np.load(episode_dir / REWARDS),
    }

def save_plot(plot_func, filepath, title, xlabel, ylabel, xticks=None, legend=True):
    plt.figure(figsize=(10, 5))
    plot_func()
    if xticks is not None:
        plt.xticks(xticks, fontsize=18)
    else:
        plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=22)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    if legend:
        plt.legend(fontsize=18, title_fontsize=18)
    plt.grid(visible=True)
    plt.tight_layout()
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')
