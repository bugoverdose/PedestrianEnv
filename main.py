import pygame
import argparse

from pedestrian_env.envs.environment import PedestrianEnv
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
    step_ms = 1000 / env.steps_per_second # default: step once every 200ms
    total_elapsed = 0
    elapsed = 0
    last_action = Action.NOTHING
    game_over = False
    game_over_info = None
    while True:
        dt = env.clock_tick()
        elapsed += dt
        total_elapsed += dt
        while elapsed >= step_ms:
            elapsed -= step_ms
            if game_over: break
            obs, reward, terminated, truncated, info = env.step(last_action)
            print(f"total_elapsed={total_elapsed}, action={last_action}, reward={reward}, cur_pos={obs['agent']}, "
                  f"cur_rewards={obs['cur_rewards']}, total_rewards={obs['total_rewards']}")
            if terminated or truncated:
                game_over = True
                game_over_info = info
            last_action = Action.NOTHING
        if game_over: break

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
    if game_over_info["is_dead"]:
        elapsed = 0
        while elapsed < 5000:
            dt = env.clock_tick()
            elapsed += dt
            env.update_positions(dt)
            env.update_time_left(total_elapsed)
            env.render()

    return False #, game_over_info["is_dead"], game_over_info["time_up"], game_over_info["game_over_score"]

def play_game(seed, max_episodes, debug):
    env = PedestrianEnv(render_mode="human", width=25, height=20, camera_size=7, steps_per_second=10, debug=debug)
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
