import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from imitation.rewards.reward_nets import RewardNet

def define_cnn_module(n_input_channels, filters_per_group, n_output_channels, kernel_size):
    padding = (kernel_size - 1) // 2  # appropriate padding for stride=1
    n_output_channels1 = n_input_channels * filters_per_group
    if len(n_output_channels) == 1:
        n_output_channels2 = n_output_channels[0]
        return nn.Sequential(
            nn.Conv2d(n_input_channels, n_output_channels1, kernel_size=kernel_size, stride=1, padding=padding, groups=n_input_channels),
            nn.ReLU(),
            nn.Conv2d(n_output_channels1, n_output_channels2, kernel_size=kernel_size, stride=1, padding=padding, groups=1),
            nn.ReLU(),
            nn.Flatten() # CNN to vector
        )
    elif len(n_output_channels) == 2:
        n_output_channels2 = n_output_channels[0]
        n_output_channels3 = n_output_channels[1]
        return nn.Sequential(
            nn.Conv2d(n_input_channels, n_output_channels1, kernel_size=kernel_size, stride=1, padding=padding, groups=n_input_channels),
            nn.ReLU(),
            nn.Conv2d(n_output_channels1, n_output_channels2, kernel_size=kernel_size, stride=1, padding=padding, groups=1),
            nn.ReLU(),
            nn.Conv2d(n_output_channels2, n_output_channels3, kernel_size=kernel_size, stride=1, padding=padding, groups=1),
            nn.ReLU(),
            nn.Flatten() # CNN to vector
        )
    else:
        raise Exception(f"{len(n_output_channels) + 1} n_output_channels not implemented")

class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 128,
                 filters_per_group = 5,
                 n_output_channels = [64, 64],
                 kernel_size=3):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = define_cnn_module(n_input_channels, filters_per_group, n_output_channels, kernel_size)

        with torch.no_grad():
            H, W = observation_space.shape[1:]
            dummy = torch.zeros(1, observation_space.shape[0], H, W)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CustomCNNRewardNet(RewardNet):
    """RewardNet that uses CNN for CHW-format float inputs (non-image channels)."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        use_state = True,
        use_action = True,
        use_next_state = False,
        use_done = False,
        filters_per_group=5,
        n_output_channels=[64, 64],
        kernel_size=3,
        mlp_hidden_size=128,
        mask_channels = [4],
    ):
        super().__init__(observation_space, action_space, normalize_images=False)

        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done

        assert isinstance(observation_space, gym.spaces.Box)
        assert len(observation_space.shape) == 3  # (C, H, W)
        assert observation_space.dtype == np.float32 or observation_space.dtype == torch.float32

        self.obs_shape = observation_space.shape
        self.act_dim = action_space.n

        # mask out to prevent reward leakage
        C = self.obs_shape[0]
        mask = torch.ones(C, dtype=torch.float32)
        for ch in mask_channels:
            mask[ch] = 0.0
        # NOTE: not updated when registered to buffer 
        self.register_buffer("channel_mask", mask.view(1, C, 1, 1))

        # CNN input channels
        in_channels = 0
        if self.use_state:
            in_channels += self.obs_shape[0]
        if self.use_next_state:
            in_channels += self.obs_shape[0]

        self.cnn = define_cnn_module(in_channels, filters_per_group, n_output_channels, kernel_size)

        # Compute CNN output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.obs_shape[1], self.obs_shape[2], device=self.channel_mask.device)
            cnn_out_dim = self.cnn(dummy).shape[1]

        mlp_input_dim = cnn_out_dim

        if self.use_action:
            mlp_input_dim += self.act_dim
        if self.use_done:
            mlp_input_dim += 1

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1)
        )

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        # self.channel_mask: (1,C,1,1)
        return x * self.channel_mask

    def forward(self, state, action, next_state, done):
        """
        Returns the estimated immediate reward for a given transition. 
        
        Used to approximate reward while training the Generator.
        
        PPO uses this trained reward to update its policy that matches expert policy
        """
        obs_list = []
        if self.use_state:
            state = self._apply_mask(state)
            obs_list.append(state)
        if self.use_next_state:
            next_state = self._apply_mask(next_state)
            obs_list.append(next_state)

        obs_cat = torch.cat(obs_list, dim=1) # (B, C or 2*C, H, W)
        cnn_out = self.cnn(obs_cat)

        inputs = [cnn_out]

        if self.use_action:
            if action.dim() == 1:
                action = action.view(-1, 1)
            elif action.dim() > 2:
                action = action.view(action.shape[0], -1)
            assert action.shape[0] == cnn_out.shape[0], f"batch mismatch: cnn {cnn_out.shape[0]} vs action {action.shape[0]}"
            inputs.append(action.float())

        if self.use_done:
            done = done.view(-1, 1).float()
            inputs.append(done)

        mlp_input = torch.cat(inputs, dim=1)
        reward = self.mlp(mlp_input)
        return reward.view(-1)
