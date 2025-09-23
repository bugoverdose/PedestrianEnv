import numpy as np

import statistics
from collections import Counter

from util import data_dir, get_sorted_episodes, load_episode_full_log

def check_episode_full_log(subjId):
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

def basic_statistics(subjId):
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    episode_score_list = []
    bonus_score_episode_cnt = 0
    run_over_episode_cnt = 0
    penalty_cnt_dict = {-100: 0, -500: 0, -1000: 0}
    actions_list = []
    movement_list = []
    for episode in episodes:
        full_log = load_episode_full_log(subj_data_dir, episode)
        episode_score_list.append(int(sum(full_log["rewards"])))

        game_end_extra_score = full_log["play_infos"][-1]["game_info"]["game_end_extra_score"]
        if game_end_extra_score > 0:
            bonus_score_episode_cnt += 1
        elif game_end_extra_score < 0:
            run_over_episode_cnt += 1
            penalty_cnt_dict[game_end_extra_score] += 1

        actions_list += list(full_log["actions"])

        movement = []
        for play_info in full_log["play_infos"]:
            cur_pos = (play_info["agent"]["grid_x"], play_info["agent"]["grid_y"])
            if len(movement) > 0 and movement[-1] == cur_pos: continue
            movement.append(cur_pos)
        movement_list.append(movement)

    total_score = sum(episode_score_list)
    best_score = max(episode_score_list)
    average_score = statistics.mean(episode_score_list)
    score_stdev = statistics.stdev(episode_score_list)
    scores = [total_score, best_score, average_score, score_stdev]

    total_episode_cnt = len(episode_score_list)
    bonus_score_episode_ratio = bonus_score_episode_cnt / total_episode_cnt
    timeover_episode_cnt = total_episode_cnt - (bonus_score_episode_cnt + run_over_episode_cnt)
    assert timeover_episode_cnt >= 0
    timeover_episode_ratio = timeover_episode_cnt / total_episode_cnt
    run_over_episode_ratio = run_over_episode_cnt / total_episode_cnt
    episode_results = [total_episode_cnt, bonus_score_episode_cnt, bonus_score_episode_ratio, timeover_episode_cnt, timeover_episode_ratio, run_over_episode_cnt, run_over_episode_ratio]

    low_penalty_cnt = penalty_cnt_dict[-100]
    low_penalty_ratio = low_penalty_cnt / run_over_episode_cnt
    low_penalty_full_ratio = low_penalty_cnt / total_episode_cnt
    mid_penalty_cnt = penalty_cnt_dict[-500]
    mid_penalty_ratio = mid_penalty_cnt / run_over_episode_cnt
    mid_penalty_full_ratio = mid_penalty_cnt / total_episode_cnt
    high_penalty_cnt = penalty_cnt_dict[-1000]
    high_penalty_ratio = high_penalty_cnt / run_over_episode_cnt
    high_penalty_full_ratio = high_penalty_cnt / total_episode_cnt
    penalties = [low_penalty_cnt, low_penalty_ratio, low_penalty_full_ratio,
                 mid_penalty_cnt, mid_penalty_ratio, mid_penalty_full_ratio,
                 high_penalty_cnt, high_penalty_ratio, high_penalty_full_ratio]

    action_counter = Counter(actions_list)
    action_proportions = {
        "Nothing": action_counter[0],
        "UP": action_counter[1],
        "DOWN": action_counter[2],
        "RIGHT": action_counter[3],
        "LEFT": action_counter[4],
    }

    visited_tile_cnt_list = []
    visited_unique_tile_cnt_list = []
    visited_tile_unique_ratio_list = []
    for movement in movement_list:
        visited_tile_cnt = len(movement)
        visited_unique_tile_cnt = len(set(movement))
        visited_tile_cnt_list.append(visited_tile_cnt)
        visited_unique_tile_cnt_list.append(visited_unique_tile_cnt)
        visited_tile_unique_ratio_list.append(visited_unique_tile_cnt / visited_tile_cnt)
    visited_tile_cnt_mean = statistics.mean(visited_tile_cnt_list)
    visited_unique_tile_cnt_mean = statistics.mean(visited_unique_tile_cnt_list)
    visited_tile_unique_ratio_mean = statistics.mean(visited_tile_unique_ratio_list)
    visited_tiles = [visited_tile_cnt_mean, visited_unique_tile_cnt_mean, visited_tile_unique_ratio_mean]

    return scores, episode_results, penalties, action_proportions, visited_tiles

if __name__ == "__main__":
    # check_episode_full_log(502)

    scores, episode_results, penalties, action_proportions, visited_tiles = basic_statistics(502)
    [total_score, best_score, average_score, score_stdev] = scores
    print(f"Total Score: {total_score}")
    print(f"Best Score: {best_score}")
    print(f"Score Mean: {average_score}")
    print(f"Score STD: {score_stdev}")
    print()

    [total_episode_cnt, bonus_score_episode_cnt, bonus_score_episode_ratio, timeover_episode_cnt, timeover_episode_ratio, run_over_episode_cnt, run_over_episode_ratio] = episode_results
    print(f"Total Episodes: {total_episode_cnt}")
    print(f"Bonus Episodes: {bonus_score_episode_cnt} (ratio={bonus_score_episode_ratio})")
    print(f"Time Over Episodes: {timeover_episode_cnt} (ratio={timeover_episode_ratio})")
    print(f"Run Over Episodes: {run_over_episode_cnt} (ratio={run_over_episode_ratio})")
    [low_penalty_cnt, low_penalty_ratio, low_penalty_full_ratio,
     mid_penalty_cnt, mid_penalty_ratio, mid_penalty_full_ratio,
     high_penalty_cnt, high_penalty_ratio, high_penalty_full_ratio] = penalties
    print(f"- Low Penalty(100): {low_penalty_cnt} (ratio={low_penalty_ratio} / full ratio={low_penalty_full_ratio})")
    print(f"- Mid Penalty(500): {mid_penalty_cnt} (ratio={mid_penalty_ratio} / full ratio={mid_penalty_full_ratio})")
    print(f"- High Penalty(1000): {high_penalty_cnt} (ratio={high_penalty_ratio} / full ratio={high_penalty_full_ratio})")
    print()

    print(f"Action Proportions: {action_proportions}")
    print()

    # Number of visited tiles per episode (e.g., exploration)
    [visited_tile_cnt_mean, visited_unique_tile_cnt_mean, visited_tile_unique_ratio_mean] = visited_tiles
    print(f"Visited Tile Count Mean: {visited_tile_cnt_mean}")
    print(f"Visited Unique Tile Count Mean: {visited_unique_tile_cnt_mean}")
    print(f"Visited Tile Unique Ratio Mean: {visited_tile_unique_ratio_mean}")

# TODO: individual level
# Crosswalk vs Jaywalking
# - (1) Crosswalk used
# - (2) Jaywalking with crosswalk in sight
# - Risk of road (e.g., car color, road size)
# - 바로 앞줄에서 플레이어를 향해 달려오는 자동차가 있음에도 앞으로 이동한 비율
# - 바로 앞칸에 아직 자동차가 남아있음에도 충분히 기다리지 못하고 앞으로 이동한 비율
