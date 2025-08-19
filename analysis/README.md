# Analysis

## Setup

Install and activate conda environment with following commands

```bash
cd analysis
conda env create -f pedestrian_analysis.yml
conda activate pedestrian_analysis
# conda env update -f pedestrian_analysis.yml
```

GPU environment setup

```bash
conda env create -f pedestrian_analysis_gpu.yml
conda activate pedestrian_analysis_gpu
# conda remove -n pedestrian_analysis_gpu --all
```

Run [TensorBoard](http://localhost:6006)

```bash
cd rl
tensorboard --logdir=./tb_logs/
```

## AIRL

`imitation` library focuses on training agents by imitating an expert,

- [Adversarial Inverse Reinforcement Learning (AIRL)](https://imitation.readthedocs.io/en/latest/algorithms/airl.html)
- [Train an Agent using Adversarial Inverse Reinforcement Learning](https://imitation.readthedocs.io/en/latest/tutorials/4_train_airl.html)


Run [TensorBoard](http://localhost:6006)

```bash
cd irl
tensorboard --logdir=./logs
```
