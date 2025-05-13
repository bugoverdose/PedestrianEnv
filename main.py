from pedestrian_env.envs.grid_world import PedestrianEnv, Actions
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
            print(f"total_elapsed={total_elapsed}, action={last_action}, reward={reward}, cur_pos={obs['agent']}")
            if terminated or truncated: return False
            last_action = Actions.nothing

        for event in pygame.event.get():
            # close window to finish early
            if event.type == pygame.QUIT: return True

            if event.type == pygame.KEYDOWN:
                last_action = KEY_ACTION.get(event.key, Actions.nothing)

        # constant rendering
        env.render()

def play_game(seed, max_episodes):
    env = PedestrianEnv(render_mode="human", size=10, steps_per_second = 5)
    for i in range(max_episodes):
        quit_game = play_episode(env, seed + i)
        if quit_game: break
    env.close()

if __name__ == "__main__":
    play_game(seed=100, max_episodes=5)
