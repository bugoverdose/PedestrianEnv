import os
import time
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

def play_episode(env, seed, verbose = False):
    obs, _ = env.reset(seed=seed)
    pygame.event.get() # clear previous key presses
    last_action = Action.NOTHING
    observations = [obs]
    actions = []
    rewards = []
    while True:
        obs, reward, terminated, truncated, info = env.step(last_action.value)
        observations.append(obs)
        actions.append(last_action.value)
        rewards.append(reward)
        if verbose:
            print(f"timeleft={env.time_left}, action={last_action}, reward={reward}, agent=({info['agent_x']}, {info['agent_y']})")
            print("Channel 0: Danger tile")
            for y in range(env.camera_height):
                print(obs[0][y])
            print("Channel 1: Crosswalk")
            for y in range(env.camera_height):
                print(obs[1][y])
            print("Channel 2: Crosswalk Activation (Crosswalk x Agent)")
            for y in range(env.camera_height):
                print(obs[2][y])
            print("Channel 3: Reachable tile")
            for y in range(env.camera_height):
                print(obs[3][y])
            print("Channel 4: Car penalty")
            for y in range(env.camera_height):
                print(obs[4][y])
            print("Channel 5: Car speed")
            for y in range(env.camera_height):
                print(obs[5][y])
            print("Channel 6: Risk level")
            for y in range(env.camera_height):
                print(obs[6][y])
            print("Channel 7: Play time left:", obs[7][0][0])
            print("Channel 8: Reward tile")
            for y in range(env.camera_height):
                print(obs[8][y])
            if len(obs) >= 10:
                print("Channel 9: Agent position")
                for y in range(env.camera_height):
                    print(obs[9][y])
        if terminated or truncated: return False, observations, actions, rewards
        last_action = Action.NOTHING

        # check if a key was being pressed down (needed for continuous movement)
        keys = pygame.key.get_pressed()
        for key, action in KEY_ACTION.items():
            if keys[key]:
                last_action = action
                break
        for event in pygame.event.get():
            # close window to finish early
            if event.type == pygame.QUIT: return True, observations, actions, rewards

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

def play_game(save_dir, session_id, seed, max_episodes, max_seconds, debug, verbose):
    episode_duration_sec = 10 if debug else 30
    env = PedestrianEnv(render_mode="human", realtime=True, gamescreen_width_fixed=False, episode_duration_sec=episode_duration_sec, debug=debug, render_sprite=True)
    # env = PedestrianEnv(render_mode="human", realtime=True, episode_duration_sec=episode_duration_sec, debug=debug, render_sprite=True)
    start_timestamp = int(time.time())
    episode_id = session_id * 1000 # assumes that each session is less than 1000 episodes
    while True:
        if max_episodes > 0 and episode_id >= max_episodes: break
        if max_seconds > 0:
            time_passed = int(time.time()) - start_timestamp
            if time_passed >= max_seconds: break
        episode_id += 1
        quit_game, observations, actions, rewards = play_episode(env, seed + episode_id, verbose)
        if quit_game: break
        os.makedirs(f"{save_dir}/{episode_id}", exist_ok=True)
        np.save(f"{save_dir}/{episode_id}/observations.npy", np.array(observations))
        np.save(f"{save_dir}/{episode_id}/actions.npy", np.array(actions))
        np.save(f"{save_dir}/{episode_id}/rewards.npy", np.array(rewards))
    env.close()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run for test")
    arg_parser.add_argument('--subjId', type=int, default=1, help='subject ID')
    arg_parser.add_argument('--sessionId', type=int, default=1, help='session ID')
    arg_parser.add_argument('--verbose', action='store_true', help='enable verbose log')
    arg_parser.add_argument('--debug', action='store_true', help='enable debugging mode')
    arg_parser.add_argument('--seed', type=int, default=1000, help='initial seed used for each episode')
    arg_parser.add_argument('--max_episodes', type=int, default=-1, help='total number of episodes')
    arg_parser.add_argument('--max_seconds', type=int, default=600, help='ends after reaching max seconds (default: 10 min)')
    args = arg_parser.parse_args()

    save_dir = f"data/{args.subjId}"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"README_{args.sessionId}.md"), 'w') as f:
        timestamp = time.time()
        dt = datetime.fromtimestamp(timestamp)
        f.write(f"Subject ID: {args.subjId}\n"
                + f"Session ID: {args.sessionId}\n"
                + f"Play start time: {dt.year}.{dt.month}.{dt.day} {dt.hour}:{dt.minute}:{dt.second}")

    play_game(save_dir=save_dir, session_id=args.sessionId, seed=args.seed, max_episodes=args.max_episodes, max_seconds=args.max_seconds, debug=args.debug, verbose=args.verbose)
