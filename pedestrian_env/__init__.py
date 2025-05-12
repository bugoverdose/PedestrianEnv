from gymnasium.envs.registration import register

register(
    id="pedestrian_env/GridWorld-v0",
    entry_point="pedestrian_env.envs:GridWorldEnv",
)
