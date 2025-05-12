from pedestrian_env.envs.grid_world import GridWorldEnv, Actions
import pygame

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
    observation, info = env.reset(seed=seed)
    while True:
        for event in pygame.event.get():
            # close window to finish early
            if event.type == pygame.QUIT: return True

            # increase step on key press
            if event.type == pygame.KEYDOWN:
                action = KEY_ACTION.get(event.key, None)
                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"action={action}, reward={reward}, done={done}")
                    if done: return False

        # constant rendering
        env.render()

def play_game(seed, max_episodes):
    env = GridWorldEnv(render_mode="human")
    for i in range(max_episodes):
        quit_game = play_episode(env, seed + i)
        if quit_game: break
    env.close()

if __name__ == "__main__":
    play_game(seed=100, max_episodes=5)
