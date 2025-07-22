# Analysis

## Setup

Install and activate conda environment with following commands

```bash
cd analysis
conda env create -f pedestrian_analysis.yml
conda activate pedestrian_analysis
```

Run [TensorBoard](http://localhost:6006)

```bash
tensorboard --logdir=./tb_logs/
```
