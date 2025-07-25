import pygame
import argparse

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
    _, _ = env.reset(seed=seed)
    pygame.event.get() # clear previous key presses
    last_action = Action.NOTHING
    while True:
        dt = env.clock_tick()
        env.elapsed += dt
        if env.elapsed < env.step_ms:
            env.apply_time_and_render(dt)
            continue
        while env.elapsed >= env.step_ms:
            obs, reward, terminated, truncated, info = env.step(last_action.value)
            if verbose:
                print(f"action={last_action}, reward={reward}, agent=({info['agent_x']}, {info['agent_y']})")
                print("Channel 0: Danger tile")
                for y in range(env.camera_size):
                    print(obs[0][y])
                print("Channel 1: Crosswalk tile")
                for y in range(env.camera_size):
                    print(obs[1][y])
                print("Channel 2: Reachable tile")
                for y in range(env.camera_size):
                    print(obs[2][y])
                print("Channel 3: Car penalty")
                for y in range(env.camera_size):
                    print(obs[3][y])
                print("Channel 4: Car speed")
                for y in range(env.camera_size):
                    print(obs[4][y])
                print("Channel 5: Risk level")
                for y in range(env.camera_size):
                    print(obs[5][y])
            if terminated or truncated: return False
            last_action = Action.NOTHING

        # check if a key was being pressed down (needed for continuous movement)
        keys = pygame.key.get_pressed()
        for key, action in KEY_ACTION.items():
            if keys[key]:
                last_action = action
                break
        for event in pygame.event.get():
            # close window to finish early
            if event.type == pygame.QUIT: return True

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

def play_game(seed, max_episodes, debug, verbose):
    episode_duration_sec = 10 if debug else 30
    env = PedestrianEnv(render_mode="human", realtime=True, episode_duration_sec=episode_duration_sec, debug=debug, render_sprite=True)
    for i in range(max_episodes):
        quit_game = play_episode(env, seed + i, verbose)
        if quit_game: break
    env.close()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run for test")
    arg_parser.add_argument('--verbose', action='store_true', help='enable verbose log')
    arg_parser.add_argument('--debug', action='store_true', help='enable debugging mode')
    arg_parser.add_argument('--seed', type=int, default=1000, help='initial seed used for each episode')
    arg_parser.add_argument('--max_episodes', type=int, default=10, help='total number of episodes')
    args = arg_parser.parse_args()

    play_game(seed=args.seed, max_episodes=args.max_episodes, debug=args.debug, verbose=args.verbose)
