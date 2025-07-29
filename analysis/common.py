import gymnasium as gym

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 kernel_size=3):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        n_output_channels = n_input_channels * 5 # 5 filters (feature maps) for each input channel
        padding = (kernel_size - 1) // 2  # appropriate padding for stride=1
        self.cnn = nn.Sequential(
            # group convolution for each channel
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=n_input_channels),
            nn.ReLU(),
            # use multiple feature map together
            nn.Conv2d(n_output_channels, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
            # nn.ReLU(),
            nn.Flatten() # CNN to vector
        )

        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
