import pandas as pd

import statistics
from collections import Counter

from util import data_dir, get_sorted_episodes, load_episode_full_log

def basic_statistics(subjIds):
    scores_dfs = []
    episode_result_dfs = []
    penalty_dfs = []
    action_proportion_dfs = []
    visited_tiles_dfs = []
    for subjId in subjIds:
        scores, episode_results, penalties, action_proportions, visited_tiles = basic_statistics_ind(subjId)
        scores_dfs.append(pd.DataFrame(scores))
        episode_result_dfs.append(pd.DataFrame(episode_results))
        penalty_dfs.append(pd.DataFrame(penalties))
        action_proportion_dfs.append(pd.DataFrame(action_proportions))
        visited_tiles_dfs.append(pd.DataFrame(visited_tiles))
    scores_df = pd.concat(scores_dfs, ignore_index=True)
    episode_result_df = pd.concat(episode_result_dfs, ignore_index=True)
    penalty_df = pd.concat(penalty_dfs, ignore_index=True)
    action_proportion_df = pd.concat(action_proportion_dfs, ignore_index=True)
    visited_tiles_df = pd.concat(visited_tiles_dfs, ignore_index=True)
    return scores_df, episode_result_df, penalty_df, action_proportion_df, visited_tiles_df

def basic_statistics_ind(subjId):
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    episode_scores_dict = {
        "full": [],
        "1": [],
        "2": [],
    }
    bonus_score_episode_cnt = 0
    run_over_episode_cnt = 0
    penalty_cnt_dict = {-100: 0, -500: 0, -1000: 0}
    actions_list = []
    movement_list = []
    for episode in episodes:
        full_log = load_episode_full_log(subj_data_dir, episode)
        episode_reward = int(sum(full_log["rewards"]))
        episode_scores_dict["full"].append(episode_reward)
        episode_scores_dict[episode[0]].append(episode_reward)

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

    scores = {
        "subj_id": [subjId],
        "total_score_full": [sum(episode_scores_dict["full"])],
        "best_score_full": [max(episode_scores_dict["full"])],
        "average_score_full": [statistics.mean(episode_scores_dict["full"])],
        "std_score_full": [statistics.stdev(episode_scores_dict["full"])],
        "total_score_session1": [sum(episode_scores_dict["1"])],
        "average_score_session1": [statistics.mean(episode_scores_dict["1"])],
        "total_score_session2": [sum(episode_scores_dict["2"])],
        "average_score_session2": [statistics.mean(episode_scores_dict["2"])],
    }
    total_episode_cnt = len(episode_scores_dict["full"])
    bonus_score_episode_ratio = bonus_score_episode_cnt / total_episode_cnt
    timeover_episode_cnt = total_episode_cnt - (bonus_score_episode_cnt + run_over_episode_cnt)
    assert timeover_episode_cnt >= 0
    timeover_episode_ratio = timeover_episode_cnt / total_episode_cnt
    run_over_episode_ratio = run_over_episode_cnt / total_episode_cnt
    episode_results = {
        "subj_id": [subjId],
        "total_episode_cnt": [total_episode_cnt], 
        "bonus_score_episode_cnt": [bonus_score_episode_cnt], 
        "bonus_score_episode_ratio": [bonus_score_episode_ratio], 
        "timeover_episode_cnt": [timeover_episode_cnt], 
        "timeover_episode_ratio": [timeover_episode_ratio], 
        "run_over_episode_cnt": [run_over_episode_cnt], 
        "run_over_episode_ratio": [run_over_episode_ratio], 
    }

    low_penalty_cnt = penalty_cnt_dict[-100]
    low_penalty_ratio = low_penalty_cnt / run_over_episode_cnt if run_over_episode_cnt > 0 else 0
    low_penalty_full_ratio = low_penalty_cnt / total_episode_cnt
    mid_penalty_cnt = penalty_cnt_dict[-500]
    mid_penalty_ratio = mid_penalty_cnt / run_over_episode_cnt if run_over_episode_cnt > 0 else 0
    mid_penalty_full_ratio = mid_penalty_cnt / total_episode_cnt
    high_penalty_cnt = penalty_cnt_dict[-1000]
    high_penalty_ratio = high_penalty_cnt / run_over_episode_cnt if run_over_episode_cnt > 0 else 0
    high_penalty_full_ratio = high_penalty_cnt / total_episode_cnt
    penalties = {
        "subj_id": [subjId],
        "low_penalty_cnt": [low_penalty_cnt],
        "low_penalty_ratio": [low_penalty_ratio],
        "low_penalty_full_ratio": [low_penalty_full_ratio],
        "mid_penalty_cnt": [mid_penalty_cnt],
        "mid_penalty_ratio": [mid_penalty_ratio],
        "mid_penalty_full_ratio": [mid_penalty_full_ratio],
        "high_penalty_cnt": [high_penalty_cnt],
        "high_penalty_ratio": [high_penalty_ratio],
        "high_penalty_full_ratio": [high_penalty_full_ratio],
    }

    action_counter = Counter(actions_list)
    action_proportions = {
        "subj_id": [subjId],
        "action_Nothing_cnt": [action_counter[0]],
        "action_UP_cnt": [action_counter[1]],
        "action_DOWN_cnt": [action_counter[2]],
        "action_RIGHT_cnt": [action_counter[3]],
        "action_LEFT_cnt": [action_counter[4]],
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
    visited_tiles = {
        "subj_id": [subjId],
        "visited_tile_cnt_mean": [visited_tile_cnt_mean], 
        "visited_unique_tile_cnt_mean": [visited_unique_tile_cnt_mean], 
        "visited_tile_unique_ratio_mean": [visited_tile_unique_ratio_mean],
    }
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
    scores, episode_results, penalties, action_proportions, visited_tiles = basic_statistics_ind(502)
    print(f"Total Score: {scores['total_score_full'][0]}")
    print(f"Best Score: {scores['best_score_full'][0]}")
    print(f"Score Mean: {scores['average_score_full'][0]}")
    print(f"Score STD: {scores['std_score_full'][0]}")
    print(f"Total Score (Session 1): {scores['total_score_session1'][0]}")
    print(f"Score Mean (Session 1): {scores['average_score_session1'][0]}")
    print(f"Total Score (Session 2): {scores['total_score_session2'][0]}")
    print(f"Score Mean (Session 2): {scores['average_score_session2'][0]}")
    print()

    print(f"Total Episodes: {episode_results['total_episode_cnt'][0]}")
    print(f"Bonus Episodes: {episode_results['bonus_score_episode_cnt'][0]} (ratio={episode_results['bonus_score_episode_ratio'][0]})")
    print(f"Time Over Episodes: {episode_results['timeover_episode_cnt'][0]} (ratio={episode_results['timeover_episode_ratio'][0]})")
    print(f"Run Over Episodes: {episode_results['run_over_episode_cnt'][0]} (ratio={episode_results['run_over_episode_ratio'][0]})")
    
    print(f"- Low Penalty(100): {penalties['low_penalty_cnt'][0]} (ratio={penalties['low_penalty_ratio'][0]} / full ratio={penalties['low_penalty_full_ratio'][0]})")
    print(f"- Mid Penalty(500): {penalties['mid_penalty_cnt'][0]} (ratio={penalties['mid_penalty_ratio'][0]} / full ratio={penalties['mid_penalty_full_ratio'][0]})")
    print(f"- High Penalty(1000): {penalties['high_penalty_cnt'][0]} (ratio={penalties['high_penalty_ratio'][0]} / full ratio={penalties['high_penalty_full_ratio'][0]})")
    print()

    print(f"Action Proportions: {action_proportions}")
    print()

    # Number of visited tiles per episode (e.g., exploration)
    print(f"Visited Tile Count Mean: {visited_tiles['visited_tile_cnt_mean'][0]}")
    print(f"Visited Unique Tile Count Mean: {visited_tiles['visited_unique_tile_cnt_mean'][0]}")
    print(f"Visited Tile Unique Ratio Mean: {visited_tiles['visited_tile_unique_ratio_mean'][0]}")

    target_road_total_counts, enter_road_with_crosswalk_counts, enter_road_without_crosswalk_counts, crosswalk_used_counts, jaywalking_counts = crosswalk_statistics(502)
    print(f"Target Road total counts: {target_road_total_counts}")
    print(f"Enter road with crosswalk: {enter_road_with_crosswalk_counts}")
    print(f"Enter road without crosswalk: {enter_road_without_crosswalk_counts}")
    print(f"Crosswalk used: {crosswalk_used_counts}")
    print(f"Jaywalking: {jaywalking_counts}")
