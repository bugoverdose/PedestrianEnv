# Project Pedestrian

## Setup

Install and activate conda environment with following commands

```bash
conda env create -f pedestrian_task_env.yml
conda activate pedestrian_task_env
```

Update dependencies or remove the created environment with following commands
```bash
conda env update -f pedestrian_task_env.yml
conda remove -n pedestrian_task_env --all
```

## Play game

Below are example commands for running the game

```bash
python run_pedestrian.py
python run_pedestrian.py --subjId=100 --sessionId=1 --max_seconds=900
python run_pedestrian.py --subjId=100 --sessionId=2 --max_seconds=900
python run_pedestrian.py --max_episodes=10 --seed=1000
python run_pedestrian.py --debug
python run_pedestrian_with_trained_model.py
```

### Subject ID

- 1~99 : development
- 100~999 : analysis test
  - 100: only cross the road using crosswalk (continue searching for crosswalk, minor mistakes)
  - 101: never cross RED without crosswalk
  - 500: PPO model simulation (optimal policy on Vectorized Env)
  - 502: PPO model simulation (optimal policy on PedestrianEnv (height=21))
  - 503: PPO model simulation (optimal policy on PedestrianEnv (height=25))
- 900 ~ 999: Pilot
- 1000 ~ : Experiment

## Run experiment

Setup environment

```bash
conda env create -f pedestrian_task_env.yml
conda env create -f behavioral_task_env.yml
```

### Run Pedestrian Task

Practice session (5 minutes)

```bash
conda activate pedestrian_task_env
python main.py --sessionId=0 --subjId=902
```

Session 1 (20 minutes)

```bash
conda activate pedestrian_task_env
python main.py --sessionId=1 --subjId=902
```

Session 2 (20 minutes)

```bash
conda activate pedestrian_task_env
python main.py --sessionId=2 --subjId=902
```

### Run Traditional Behavioral Tasks

DDT (ADO version)

```bash
conda activate behavioral_task_env
python behavioral_tasks/ddt/ddt.py --subjId=902
```

CRA (ADO version)

```bash
conda activate behavioral_task_env
python behavioral_tasks/cra/cra.py --subjId=902
```
