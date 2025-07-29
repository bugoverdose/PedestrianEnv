import numpy as np

import torch as th
import torch.nn as nn
from imitation.rewards.reward_nets import RewardNet
from gymnasium import spaces

class CustomCNNRewardNet(RewardNet):
    """RewardNet that uses CNN for CHW-format float inputs (non-image channels)."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        kernel_size=3,
        hid_channels=(32, 32),
        mlp_hidden_size=128,
    ):
        super().__init__(observation_space, action_space)

        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done

        assert isinstance(observation_space, spaces.Box)
        assert len(observation_space.shape) == 3  # (C, H, W)
        assert observation_space.dtype == np.float32 or observation_space.dtype == th.float32

        self.obs_shape = observation_space.shape
        self.act_dim = action_space.n

        # CNN input channels
        in_channels = 0
        if self.use_state:
            in_channels += self.obs_shape[0]
        if self.use_next_state:
            in_channels += self.obs_shape[0]

        # CNN module
        padding = (kernel_size - 1) // 2
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels[0], kernel_size=kernel_size, stride=1,padding=padding),
            nn.ReLU(),
            nn.Conv2d(hid_channels[0], hid_channels[1], kernel_size=kernel_size, stride=1,padding=padding),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute CNN output size dynamically
        with th.no_grad():
            dummy = th.zeros(1, in_channels, self.obs_shape[1], self.obs_shape[2])
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

    def forward(self, state, action, next_state, done):
        inputs = []

        obs_list = []
        if self.use_state:
            obs_list.append(state)
        if self.use_next_state:
            obs_list.append(next_state)
        obs_cat = th.cat(obs_list, dim=1)  # concatenate along channel axis
        cnn_out = self.cnn(obs_cat)
        inputs.append(cnn_out)

        if self.use_action:
            # one-hot encode discrete action
            action_oh = nn.functional.one_hot(action, num_classes=self.act_dim).float()
            inputs.append(action_oh)

        if self.use_done:
            done = done.view(-1, 1).float()
            inputs.append(done)

        mlp_input = th.cat(inputs, dim=1)
        reward = self.mlp(mlp_input)
        return reward.view(-1)

# class CustomCNNExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim=256):
#         super().__init__(observation_space, features_dim)
#         C, H, W = observation_space.shape
#         n_output_channels = C * 5
#         kernel_size = 3
#         padding = (kernel_size - 1) // 2
#         self.cnn = nn.Sequential(
#             nn.Conv2d(C, n_output_channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, C, H, W)
#             n_flatten = self.cnn(dummy_input).shape[1]
#         self._features_dim = n_flatten

#     def forward(self, obs):
#         return self.cnn(obs)
