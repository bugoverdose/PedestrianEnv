import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from analysis.util import data_dir, get_sorted_episodes, load_episode_play_log

if __name__ == "__main__":
    subjId = 1
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    for episode in episodes:
        play_log = load_episode_play_log(subj_data_dir, episode)
        print(subj_data_dir / episode)
        print(play_log["road_metadata"][0])
        print(play_log["car_metadata"][0])
        print(play_log["observations"][0][0][0])
        print(play_log["play_infos"][0])
        print(play_log["car_infos"][0])
        print(play_log["actions"][0])
        print(play_log["rewards"][0])
