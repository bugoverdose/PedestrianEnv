import numpy as np

# import math
import statistics

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
    penalty_cnt_dict = {
        -100: 0,
        -500: 0,
        -1000: 0,
    }

    for episode in episodes:
        full_log = load_episode_full_log(subj_data_dir, episode)
        episode_score_list.append(int(sum(full_log["rewards"])))

        game_end_extra_score = full_log["play_infos"][-1]["game_info"]["game_end_extra_score"]
        if game_end_extra_score > 0:
            bonus_score_episode_cnt += 1
        elif game_end_extra_score < 0:
            run_over_episode_cnt += 1
            penalty_cnt_dict[game_end_extra_score] += 1

        # print()
        # print(subj_data_dir / episode)
        # print("env_configuration")
        # print(full_log["episode_metadata"]["env_configuration"])
        # print("episode_configuration")
        # print(full_log["episode_metadata"]["episode_configuration"])
        # print("play_infos")
        # print(full_log["play_infos"][0])
        # print("observations")
        # print(full_log["observations"][0])
        # print("observation size:", len(full_log["observations"][0]))
        # print("actions:", full_log["actions"][:10])
        # print("rewards:", full_log["rewards"][:10])
        # assert type(full_log["rewards"][0]) == np.float32 or type(full_log["rewards"][0]) == np.float64
        # print("trajectory size:", len(full_log["actions"]))
        # assert len(full_log["play_infos"]) == len(full_log["observations"])
        # assert len(full_log["actions"]) == len(full_log["rewards"])
        # assert len(full_log["observations"]) == len(full_log["actions"]) + 1

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
    return scores, episode_results, penalties

if __name__ == "__main__":
    # check_episode_full_log(502)

    scores, episode_results, penalties = basic_statistics(502)
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


# TODO: individual level
# Crosswalk vs Jaywalking
# - (1) Crosswalk used
# - (2) Jaywalking with crosswalk in sight
# - Risk of road (e.g., car color, road size)
# - 바로 앞줄에서 플레이어를 향해 달려오는 자동차가 있음에도 앞으로 이동한 비율
# - 바로 앞칸에 아직 자동차가 남아있음에도 충분히 기다리지 못하고 앞으로 이동한 비율

# Action proportions (No action vs Up vs Down vs Left/Right)
# Number of visited tiles per episode (e.g., exploration)
