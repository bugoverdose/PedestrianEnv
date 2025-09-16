from util import data_dir, get_sorted_episodes, load_episode_full_log

def check_episode_full_log(subjId):
    subj_data_dir = data_dir(subjId)
    episodes = get_sorted_episodes(subj_data_dir)
    for episode in episodes:
        full_log = load_episode_full_log(subj_data_dir, episode)
        print()
        print(subj_data_dir / episode)
        print("env_configuration")
        print(full_log["env_configuration"])
        print("episode_configuration")
        print(full_log["episode_configuration"])
        print("play_infos")
        print(full_log["play_infos"][0])
        print("observations")
        print(full_log["observations"][0])
        print("observation size:", len(full_log["observations"][0]))
        print("actions:", full_log["actions"][:10])
        print("rewards:", full_log["rewards"][:10])
        print("trajectory size:", len(full_log["observations"]))

if __name__ == "__main__":
    check_episode_full_log(1)

# TODO: 분석 함수 구현 필요
# - 바로 앞줄에서 플레이어를 향해 달려오는 자동차가 있음에도 앞으로 이동한 비율
# - 바로 앞칸에 아직 자동차가 남아있음에도 충분히 기다리지 못하고 앞으로 이동한 비율
