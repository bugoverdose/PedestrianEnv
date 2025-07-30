import gymnasium as gym

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 filters_per_group = 3,
                 n_output_channels = [32],
                 kernel_size=3):
        super().__init__(observation_space, features_dim)

        padding = (kernel_size - 1) // 2  # appropriate padding for stride=1
        n_input_channels = observation_space.shape[0]
        n_output_channels1 = n_input_channels * filters_per_group
        if len(n_output_channels) == 1:
            n_output_channels2 = n_output_channels[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, n_output_channels1, kernel_size=kernel_size, stride=1, padding=padding, groups=n_input_channels),
                nn.ReLU(),
                nn.Conv2d(n_output_channels1, n_output_channels2, kernel_size=kernel_size, stride=1, padding=padding, groups=1),
                nn.ReLU(),
                nn.Flatten() # CNN to vector
            )
        elif len(n_output_channels) == 2:
            n_output_channels2 = n_output_channels[0]
            n_output_channels3 = n_output_channels[1]
            self.cnn = nn.Sequential(
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

        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
