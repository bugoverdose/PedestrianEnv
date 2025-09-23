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

def crosswalk_statistics(subjId):
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    # all the target roads that the player started to cross (includes getting hit by car)
    target_road_total_counts = {"total": 0, "easy": 0, "hard": 0, "low_penalty": 0, "mid_penalty": 0, "high_penalty": 0}

    # started crossing the road after activating it
    enter_road_with_crosswalk_counts = {"total": 0, "easy": 0, "hard": 0, "low_penalty": 0, "mid_penalty": 0, "high_penalty": 0}

    # started crossing the road without using the crosswalk in sight (includes getting hit by car)
    enter_road_without_crosswalk_counts = {"total": 0, "easy": 0, "hard": 0, "low_penalty": 0, "mid_penalty": 0, "high_penalty": 0}

    # activated the crosswalk and reached the end of the road
    crosswalk_used_counts = {"total": 0, "easy": 0, "hard": 0, "low_penalty": 0, "mid_penalty": 0, "high_penalty": 0}

    # never used the crosswalk that is in sight
    jaywalking_counts = {"total": 0, "easy": 0, "hard": 0, "low_penalty": 0, "mid_penalty": 0, "high_penalty": 0}

    for episode in episodes:
        full_log = load_episode_full_log(subj_data_dir, episode)

        road_dict = {}
        for road in full_log["episode_metadata"]["episode_configuration"]["roads"]["roads"]:
            road_size = road["bottom_y"] - road["top_y"]
            assert road_size == 1 or road_size == 3

            risk = "high_penalty"
            if road["penalty"] == 100:
                risk = "low_penalty"
            elif road["penalty"] == 500:
                risk = "mid_penalty"

            road_dict[road["uid"]] = {
                "difficulty": "easy" if road_size == 1 else "hard",
                "risk": risk, 
                "crosswalk": road["crosswalk"], # {"uid": road.crosswalk.uid, "col": road.crosswalk.col}
            }

        entered_road_uid_set = set()
        crossed_road_uid_set = set()
        last_crossing_road_info = None
        for i in range(len(full_log["actions"])):
            play_info = full_log["play_infos"][i]
            cur_pos = (play_info["agent"]["grid_x"], play_info["agent"]["grid_y"])
            action = full_log["actions"][i]
            is_up = action == 1
            crosswalk_activated = play_info["map"]["activated_crosswalk_uid"] != 0

            target_road_uid = play_info["map"]["target_road_uid"]
            road_difficulty = road_dict[target_road_uid]["difficulty"]
            road_risk = road_dict[target_road_uid]["risk"]
            keys = ["total", road_difficulty, road_risk]

            is_danger_zone = full_log["episode_metadata"]["episode_configuration"]["roads"]["is_dangerous_row"]
            cur_safe_zone = not is_danger_zone[cur_pos[1]]
            next_safe_zone = not is_danger_zone[cur_pos[1]-1]
            if is_up and cur_safe_zone:
                # left the safe zone and just started crossing the target road
                entered_road_with_crosswalk_visible = play_info["map"]["target_road_crosswalk_visible"]
                if target_road_uid not in entered_road_uid_set:
                    entered_road_uid_set.add(target_road_uid)
                    # first time entering the target road
                    for key in keys:
                        target_road_total_counts[key] += 1
                        if crosswalk_activated:
                            enter_road_with_crosswalk_counts[key] += 1
                        else:
                            if entered_road_with_crosswalk_visible:
                                enter_road_without_crosswalk_counts[key] += 1
                # restart saving the trajectory every time the player enters the road until the first successful crossing
                if target_road_uid not in crossed_road_uid_set:
                    last_crossing_road_info = {
                        "uid": target_road_uid,
                        "trajectory": [cur_pos],
                        "crosswalk_always_activated": crosswalk_activated,
                        "entered_road_with_crosswalk_visible": entered_road_with_crosswalk_visible,
                    }

            if last_crossing_road_info is not None:
                if target_road_uid == last_crossing_road_info["uid"]:
                    if last_crossing_road_info["trajectory"][-1] != cur_pos:
                        last_crossing_road_info["trajectory"].append(cur_pos)
                        if not crosswalk_activated:
                            last_crossing_road_info["crosswalk_always_activated"] = False
                    if is_up and next_safe_zone:
                        if target_road_uid not in crossed_road_uid_set:
                            # finished crossing the road and entered the safe zone for the first time
                            crossed_road_uid_set.add(target_road_uid)
                            for key in keys:
                                target_road_total_counts[key] += 1
                                if last_crossing_road_info["crosswalk_always_activated"]:
                                    crosswalk_used_counts[key] += 1
                                if last_crossing_road_info["entered_road_with_crosswalk_visible"]:
                                    jaywalking_counts[key] += 1
                else:
                    # went back to previously crossed road. should be ignored from the statistics.
                    last_crossing_road_info = None

    return target_road_total_counts, enter_road_with_crosswalk_counts, enter_road_without_crosswalk_counts, crosswalk_used_counts, jaywalking_counts

# TODO: 바로 앞줄에서 플레이어를 향해 달려오는 자동차가 있음에도 앞으로 이동한 비율
# TODO: 바로 앞칸에 아직 자동차가 남아있음에도 충분히 기다리지 못하고 앞으로 이동한 비율
def move_up_statistics(subjId):
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    for episode in episodes:
        full_log = load_episode_full_log(subj_data_dir, episode)

        for i in range(len(full_log["actions"])):
            action = full_log["actions"][i]
            is_up = action == 1
            if not is_up: continue
            play_info = full_log["play_infos"][i]

if __name__ == "__main__":
    # check_episode_full_log(502)

    # scores, episode_results, penalties, action_proportions, visited_tiles = basic_statistics(502)
    # [total_score, best_score, average_score, score_stdev] = scores
    # print(f"Total Score: {total_score}")
    # print(f"Best Score: {best_score}")
    # print(f"Score Mean: {average_score}")
    # print(f"Score STD: {score_stdev}")
    # print()

    # [total_episode_cnt, bonus_score_episode_cnt, bonus_score_episode_ratio, timeover_episode_cnt, timeover_episode_ratio, run_over_episode_cnt, run_over_episode_ratio] = episode_results
    # print(f"Total Episodes: {total_episode_cnt}")
    # print(f"Bonus Episodes: {bonus_score_episode_cnt} (ratio={bonus_score_episode_ratio})")
    # print(f"Time Over Episodes: {timeover_episode_cnt} (ratio={timeover_episode_ratio})")
    # print(f"Run Over Episodes: {run_over_episode_cnt} (ratio={run_over_episode_ratio})")
    # [low_penalty_cnt, low_penalty_ratio, low_penalty_full_ratio,
    #  mid_penalty_cnt, mid_penalty_ratio, mid_penalty_full_ratio,
    #  high_penalty_cnt, high_penalty_ratio, high_penalty_full_ratio] = penalties
    # print(f"- Low Penalty(100): {low_penalty_cnt} (ratio={low_penalty_ratio} / full ratio={low_penalty_full_ratio})")
    # print(f"- Mid Penalty(500): {mid_penalty_cnt} (ratio={mid_penalty_ratio} / full ratio={mid_penalty_full_ratio})")
    # print(f"- High Penalty(1000): {high_penalty_cnt} (ratio={high_penalty_ratio} / full ratio={high_penalty_full_ratio})")
    # print()

    # print(f"Action Proportions: {action_proportions}")
    # print()

    # # Number of visited tiles per episode (e.g., exploration)
    # [visited_tile_cnt_mean, visited_unique_tile_cnt_mean, visited_tile_unique_ratio_mean] = visited_tiles
    # print(f"Visited Tile Count Mean: {visited_tile_cnt_mean}")
    # print(f"Visited Unique Tile Count Mean: {visited_unique_tile_cnt_mean}")
    # print(f"Visited Tile Unique Ratio Mean: {visited_tile_unique_ratio_mean}")

    target_road_total_counts, enter_road_with_crosswalk_counts, enter_road_without_crosswalk_counts, crosswalk_used_counts, jaywalking_counts = crosswalk_statistics(502)
    print(f"Target Road total counts: {target_road_total_counts}")
    print(f"Enter road with crosswalk: {enter_road_with_crosswalk_counts}")
    print(f"Enter road without crosswalk: {enter_road_without_crosswalk_counts}")
    print(f"Crosswalk used: {crosswalk_used_counts}")
    print(f"Jaywalking: {jaywalking_counts}")
