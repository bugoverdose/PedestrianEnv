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
python main.py
python main.py --max_episodes=10 --seed=1000
python main.py --debug
```
