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

def play_episode(env, seed):
    _, _ = env.reset(seed=seed)
    total_elapsed = 0
    elapsed = 0
    pygame.event.get() # clear previous key presses
    last_action = Action.NOTHING
    while True:
        dt = env.clock_tick()
        elapsed += dt
        total_elapsed += dt
        while elapsed >= env.step_ms:
            elapsed -= env.step_ms
            obs, reward, terminated, truncated, info = env.step(last_action)
            print(f"time_left={obs[0]:.0f}, " +
                  f"action={last_action}, reward={reward}, cur_pos=({obs[1], obs[2]})" 
                  # f"distance until next road = ({obs[3], obs[4]}), " +
            )
            if terminated or truncated:
                env.render_game_over()
                return False
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
            # else:
            #     # check if stopped to press a key (needed for instant stop)
            #     if event.type == pygame.KEYUP and KEY_ACTION.get(event.key) == last_action:
            #         last_action = Actions.nothing
            #         print("KEYUP")
            #         break

        # constant update and rendering
        env.update_positions(dt)
        env.update_time_left(total_elapsed)
        env.render()

def play_game(seed, max_episodes, debug):
    episode_duration_sec = 10 if debug else 30
    env = PedestrianEnv(render_mode="human", episode_duration_sec=episode_duration_sec, debug=debug, render_sprite=True)
    for i in range(max_episodes):
        quit_game = play_episode(env, seed + i)
        if quit_game: break
    env.close()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run for test")
    arg_parser.add_argument('--debug', action='store_true', help='enable debugging mode')
    arg_parser.add_argument('--seed', type=int, default=1000, help='initial seed used for each episode')
    arg_parser.add_argument('--max_episodes', type=int, default=10, help='total number of episodes')
    args = arg_parser.parse_args()

    play_game(seed=args.seed, max_episodes=args.max_episodes, debug=args.debug)
