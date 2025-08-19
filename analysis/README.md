# Analysis

## Setup

Install and activate conda environment with following commands

```bash
cd analysis
conda env create -f pedestrian_analysis.yml
conda activate pedestrian_analysis
# conda env update -f pedestrian_analysis.yml
```

CCSL3 setup

```bash
conda env create -f pedestrian_analysis_gpu.yml
conda activate pedestrian_analysis_gpu
# conda remove -n pedestrian_analysis_gpu --all
```

Run [TensorBoard](http://localhost:6006)

```bash
tensorboard --logdir=./tb_logs/
```
