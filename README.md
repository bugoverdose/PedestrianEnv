# Project Pedestrian

## Setup

Install and activate conda environment with following commands

```bash
conda env create -f pedestrian_game.yml
conda activate pedestrian_game
```

Update dependencies or remove the created environment with following commands
```bash
conda env update -f pedestrian_game.yml
conda remove -n pedestrian_game --all
```

## Play game

Below are example commands for running the game

```bash
python run_pedestrian.py
python run_pedestrian.py --subjId=100 --sessionId=1 --max_seconds=900
python run_pedestrian.py --subjId=100 --sessionId=2 --max_seconds=900
python run_pedestrian.py --max_episodes=10 --seed=1000
python run_pedestrian.py --debug
```

## Run experiment

Setup and activate environment

```bash
conda env create -f experiment_env.yml
conda activate pedestrian_experiment
```

Run Pedestrian Task

```bash
# practice session: 5 minutes
python main.py --subjId=1001 --sessionId=0

# main session 1: 15 minutes
python main.py --subjId=1001 --sessionId=1

# Rest Time: 5 minutes

# main session 2: 15 minutes
python main.py --subjId=1001 --sessionId=2
```

Run Behavioral Tasks

```bash
# CRA + ADO
python behavioral_tasks/cra/cra.py --subjId=1001

# Rest Time: 5 minutes

# DDT + ADO
python behavioral_tasks/ddt/ddt.py --subjId=1001
```
