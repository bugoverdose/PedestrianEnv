import pygame
import argparse

from pedestrian_env.envs.environment import PedestrianEnv
from pedestrian_env.envs.action import Action

KEY_ACTION = {
    pygame.K_UP: Action.up,
    pygame.K_DOWN: Action.down,
    pygame.K_RIGHT: Action.right,
    pygame.K_LEFT: Action.left,

    pygame.K_w: Action.up,
    pygame.K_s: Action.down,
    pygame.K_d: Action.right,
    pygame.K_a: Action.left,
}

def play_episode(env, seed):
    _, _ = env.reset(seed=seed)
    step_ms = 1000 / env.steps_per_second # default: step once every 200ms
    total_elapsed = 0
    elapsed = 0
    last_action = Action.nothing
    while True:
        dt = env.clock_tick()
        elapsed += dt
        total_elapsed += dt
        while elapsed >= step_ms:
            elapsed -= step_ms
            obs, reward, terminated, truncated, info = env.step(last_action)
            print(f"total_elapsed={total_elapsed}, action={last_action}, reward={reward}, cur_pos={obs['agent']}, "
                  f"cur_rewards={obs['cur_rewards']}, total_rewards={obs['total_rewards']}")
            if terminated or truncated: return False
            last_action = Action.nothing

        # check if a key was being pressed down (needed for continuous movement)
        keys = pygame.key.get_pressed()
        for key, action in KEY_ACTION.items():
            if keys[key]:
                last_action = action
                break
        for event in pygame.event.get():
            # close window to finish early
            if event.type == pygame.QUIT: return True

            if last_action == Action.nothing:
                # check if started to press a key (needed for instant start)
                if event.type == pygame.KEYDOWN:
                    last_action = KEY_ACTION.get(event.key, Action.nothing)
                    if last_action != Action.nothing: break
            # else:
            #     # check if stopped to press a key (needed for instant stop)
            #     if event.type == pygame.KEYUP and KEY_ACTION.get(event.key) == last_action:
            #         last_action = Actions.nothing
            #         print("KEYUP")
            #         break

        # constant update and rendering
        env.update_positions(dt)
        env.render()

def play_game(seed, max_episodes):
    env = PedestrianEnv(render_mode="human", width=10, height=20, camera_size=7, steps_per_second=10)
    for i in range(max_episodes):
        quit_game = play_episode(env, seed + i)
        if quit_game: break
    env.close()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run for test")
    arg_parser.add_argument('--seed', type=int, default=1000, help='initial seed used for each episode')
    arg_parser.add_argument('--max_episodes', type=int, default=10, help='total number of episodes')
    args = arg_parser.parse_args()

    play_game(seed=args.seed, max_episodes=args.max_episodes)
