import torch
import numpy as np
import gymnasium as gym

from imitation.data.types import Trajectory
from typing import List, Dict, Any

# # === 1) 저장본 로드 ===
# ckpt = torch.load(f"{save_dir}reward_net.pt", map_location="cpu")
# obs_space = ckpt["obs_space"]
# action_space = ckpt["action_space"]

# # 학습 때 사용한 RewardNet과 같은 클래스/인자 필요!
# from imitation.rewards.reward_nets import BasicRewardNet
# reward_net = BasicRewardNet(
#     observation_space=obs_space,
#     action_space=action_space,
#     # 학습 시 설정과 반드시 동일해야 함
#     use_state=True, use_action=True, use_next_state=False, use_done=False,
#     # 예: hid_sizes=(256,256) 등… (학습 때와 동일)
# )
# reward_net.load_state_dict(ckpt["model_state_dict"], strict=True)
# reward_net.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# reward_net.to(device)

if __name__ == "__main__":
    ckpt = torch.load(f"{save_dir}reward_net.pt", map_location="cpu")
    obs_space = ckpt["obs_space"]
    action_space = ckpt["action_space"]

    # 학습 때 사용한 RewardNet과 같은 클래스/인자 필요!
    from imitation.rewards.reward_nets import BasicRewardNet
    reward_net = BasicRewardNet(
        observation_space=obs_space,
        action_space=action_space,
        # 학습 시 설정과 반드시 동일해야 함
        use_state=True, use_action=True, use_next_state=False, use_done=False,
        # 예: hid_sizes=(256,256) 등… (학습 때와 동일)
    )
    reward_net.load_state_dict(ckpt["model_state_dict"], strict=True)
    reward_net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_net.to(device)

    # full_build_mlp_kwargs: Dict[str, Any] = {
    #     "hid_sizes": (32, 32),
    #     "hid_sizes": (32, 32, 64),
    #     "in_size": 100,
    #     "out_size": 1,
    #     "squeeze_output": True,
    # }
    # def a(
    #     in_size,
    #     hid_sizes,
    #     out_size,
    #     squeeze_output,
    # ):
    #     print(hid_sizes)
    # a(**full_build_mlp_kwargs)
