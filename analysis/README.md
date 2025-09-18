# Analysis

## Setup

Install and activate conda environment with following commands

```bash
cd analysis
conda env create -f pedestrian_analysis.yml
# or mamba env create -f pedestrian_analysis.yml
conda activate pedestrian_analysis
# conda env update -f pedestrian_analysis.yml
```

GPU environment setup

```bash
conda env create -f pedestrian_analysis_gpu.yml
conda activate pedestrian_analysis_gpu
# conda remove -n pedestrian_analysis_gpu --all
```

## AIRL

`imitation` library focuses on training agents by imitating an expert.

- [Adversarial Inverse Reinforcement Learning (AIRL)](https://imitation.readthedocs.io/en/latest/algorithms/airl.html)
- [Train an Agent using Adversarial Inverse Reinforcement Learning](https://imitation.readthedocs.io/en/latest/tutorials/4_train_airl.html)
- [Reward Networks](https://imitation.readthedocs.io/en/latest/main-concepts/reward_networks.html)

## Tensorboard

The training process for each model can be analyzed using [TensorBoard](http://localhost:6006).

```bash
tensorboard --logdir=./rl/tb_logs
tensorboard --logdir=./irl/tb_logs
```
