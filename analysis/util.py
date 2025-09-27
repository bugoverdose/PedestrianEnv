import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

EPISODE_METADATA = "episode_metadata.npy"
PLAY_INFOS = "play_infos.npy"
OBSERVATIONS = "observations.npy"
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

def load_episode_full_log(subj_data_dir, episode):
    episode_dir = subj_data_dir / episode
    episode_metadata = np.load(episode_dir / EPISODE_METADATA, allow_pickle=True)[0]
    return {
        "episode_metadata": episode_metadata,
        "observations": np.load(episode_dir / OBSERVATIONS),
        "play_infos": np.load(episode_dir / PLAY_INFOS, allow_pickle=True),
        "actions": np.load(episode_dir / ACTIONS),
        "rewards": np.load(episode_dir / REWARDS).astype(np.float32),
    }

# def save_plot(plot_func, filepath, title, xlabel, ylabel, xticks=None, legend=True):
#     plt.figure(figsize=(10, 5))
#     plot_func()
#     if xticks is not None:
#         plt.xticks(xticks, fontsize=18)
#     else:
#         plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title(title, fontsize=22)
#     plt.xlabel(xlabel, fontsize=20)
#     plt.ylabel(ylabel, fontsize=20)
#     if legend:
#         plt.legend(fontsize=18, title_fontsize=18)
#     plt.grid(visible=True)
#     plt.tight_layout()
#     if filepath is None:
#         plt.show()
#     else:
#         plt.savefig(filepath, bbox_inches='tight')

if __name__ == "__main__":
    subjId = 100
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    for episode in episodes:
        full_log = load_episode_full_log(subj_data_dir, episode)
        print()
        print(subj_data_dir / episode)
        print("env_configuration")
        print(full_log["episode_metadata"]["env_configuration"])
        print("episode_configuration")
        print(full_log["episode_metadata"]["episode_configuration"])
        print("play_infos")
        print(full_log["play_infos"][0])
        print("observations")
        print(full_log["observations"][0])
        print("observation size:", len(full_log["observations"][0]))
        print("actions:", full_log["actions"][:10])
        print("rewards:", full_log["rewards"][:10])
        assert type(full_log["rewards"][0]) == np.float32 or type(full_log["rewards"][0]) == np.float64
        print("trajectory size:", len(full_log["actions"]))
        assert len(full_log["play_infos"]) == len(full_log["observations"])
        assert len(full_log["actions"]) == len(full_log["rewards"])
        assert len(full_log["observations"]) == len(full_log["actions"]) + 1
