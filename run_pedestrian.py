import os
import time
import json
from datetime import datetime

import pygame
import argparse
import numpy as np

from pedestrian_env.envs import PedestrianEnv
from pedestrian_env.envs.action import Action

KEY_ACTION = {
    pygame.K_UP: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_LEFT: Action.LEFT,

    pygame.K_w: Action.UP,
    pygame.K_s: Action.DOWN,
    pygame.K_d: Action.RIGHT,
    pygame.K_a: Action.LEFT,
}

def play_episode(env, episode_seed, verbose = False):
    obs, info = env.reset(seed=episode_seed)
    observations = [obs]
    episode_metadata = { "road_metadata": info["road_metadata"], "car_metadata": info["car_metadata"] }
    play_infos = [info["play_infos"]]
    car_infos = [info["cars"]]
    actions = []
    rewards = []

    pygame.event.get() # clear previous key presses
    last_action = Action.NOTHING
    while True:
        obs, reward, terminated, truncated, info = env.step(last_action.value)
        observations.append(obs)
        play_infos.append(info["play_infos"])
        car_infos.append(info["cars"])
        actions.append(last_action.value)
        rewards.append(reward)
        if verbose:
            agent_x = info["play_infos"][0]
            agent_y = info["play_infos"][1]
            print(f"timeleft={env.time_left}, action={last_action}, reward={reward}, agent=({agent_x}, {agent_y})")
        if terminated or truncated: return False, episode_metadata, observations, actions, rewards, play_infos, car_infos
        last_action = Action.NOTHING

        # check if a key was being pressed down (needed for continuous movement)
        keys = pygame.key.get_pressed()
        for key, action in KEY_ACTION.items():
            if keys[key]:
                last_action = action
                break
        for event in pygame.event.get():
            # close window to finish early
            if event.type == pygame.QUIT: return True, episode_metadata, observations, actions, rewards, play_infos, car_infos

            if last_action == Action.NOTHING:
                # check if started to press a key (needed for instant start)
                if event.type == pygame.KEYDOWN:
                    last_action = KEY_ACTION.get(event.key, Action.NOTHING)
                    if last_action != Action.NOTHING: break
            else:
                # check if stopped to press a key (needed for instant stop)
                if event.type == pygame.KEYUP and KEY_ACTION.get(event.key) == last_action:
                    last_action = Action.NOTHING
                    break

def play_game(base_dir, session_id, max_seconds, max_episodes=-1, base_seed=0, debug=False, verbose=False):
    episode_duration_sec = 10 if debug else 30
    env = PedestrianEnv(render_mode="human", realtime=True, episode_duration_sec=episode_duration_sec, debug=debug)
    start_timestamp = int(time.time())
    episode_id = session_id * 1000 # assumes that each session is less than 1000 episodes
    while True:
        if max_episodes > 0 and episode_id >= max_episodes: break
        if max_seconds > 0:
            time_passed = int(time.time()) - start_timestamp
            if time_passed >= max_seconds: break
        episode_id += 1
        episode_seed = base_seed + episode_id
        quit_game, episode_metadata, observations, actions, rewards, play_infos, car_infos = play_episode(env, episode_seed, verbose)
        if quit_game: break
        # NOTE: .npy is more lightweight than CSV because it stores data in pure binary format
        os.makedirs(f"{base_dir}/{episode_seed:04d}", exist_ok=True)
        with open(f"{base_dir}/{episode_seed:04d}/episode_metadata.json", "w") as f:
            json.dump(episode_metadata, f)
        np.save(f"{base_dir}/{episode_seed:04d}/observations.npy", np.array(observations))
        np.save(f"{base_dir}/{episode_seed:04d}/actions.npy", np.array(actions))
        np.save(f"{base_dir}/{episode_seed:04d}/rewards.npy", np.array(rewards))
        np.save(f"{base_dir}/{episode_seed:04d}/play_infos.npy", np.array(play_infos, dtype=object), allow_pickle=True)
        np.save(f"{base_dir}/{episode_seed:04d}/car_infos.npy", np.array(car_infos, dtype=object), allow_pickle=True)
    env.close()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run for test")
    arg_parser.add_argument('--subjId', type=int, default=1, help='subject ID')
    arg_parser.add_argument('--sessionId', type=int, default=1, help='session ID')
    arg_parser.add_argument('--verbose', action='store_true', help='enable verbose log')
    arg_parser.add_argument('--debug', action='store_true', help='enable debugging mode')
    arg_parser.add_argument('--seed', type=int, default=0, help='seed value added')
    arg_parser.add_argument('--max_episodes', type=int, default=-1, help='total number of episodes')
    arg_parser.add_argument('--max_seconds', type=int, default=600, help='ends after reaching max seconds (default: 10 min)')
    args = arg_parser.parse_args()

    base_dir = f"data/{args.subjId}"
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, f"README_{args.sessionId}.md"), 'w') as f:
        timestamp = time.time()
        dt = datetime.fromtimestamp(timestamp)
        f.write(f"Subject ID: {args.subjId}\n"
                + f"Session ID: {args.sessionId}\n"
                + f"Play start time: {dt.year}.{dt.month}.{dt.day} {dt.hour}:{dt.minute}:{dt.second}")

    play_game(base_dir=base_dir, session_id=args.sessionId, base_seed=args.seed, max_episodes=args.max_episodes, max_seconds=args.max_seconds, debug=args.debug, verbose=args.verbose)
