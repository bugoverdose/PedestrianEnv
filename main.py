import pygame
import argparse

from pedestrian_env.envs.environment import PedestrianEnv, Actions

KEY_ACTION = {
    pygame.K_UP: Actions.up,
    pygame.K_DOWN: Actions.down,
    pygame.K_RIGHT: Actions.right,
    pygame.K_LEFT: Actions.left,

    pygame.K_w: Actions.up,
    pygame.K_s: Actions.down,
    pygame.K_d: Actions.right,
    pygame.K_a: Actions.left,
}

def play_episode(env, seed):
    _, _ = env.reset(seed=seed)
    step_ms = 1000 / env.steps_per_second # default: step once every 200ms
    total_elapsed = 0
    elapsed = 0
    last_action = Actions.nothing
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
            last_action = Actions.nothing

        # check if a key was being pressed down (needed for continuous movement)
        keys = pygame.key.get_pressed()
        for key, action in KEY_ACTION.items():
            if keys[key]:
                last_action = action
                break
        for event in pygame.event.get():
            # close window to finish early
            if event.type == pygame.QUIT: return True

            if last_action == Actions.nothing:
                # check if started to press a key (needed for instant start)
                if event.type == pygame.KEYDOWN:
                    last_action = KEY_ACTION.get(event.key, Actions.nothing)
                    if last_action != Actions.nothing: break
            # else:
            #     # check if stopped to press a key (needed for instant stop)
            #     if event.type == pygame.KEYUP and KEY_ACTION.get(event.key) == last_action:
            #         last_action = Actions.nothing
            #         print("KEYUP")
            #         break

        # constant rendering
        env.render()

def play_game(seed, max_episodes):
    env = PedestrianEnv(render_mode="human", size=10, steps_per_second = 5)
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
