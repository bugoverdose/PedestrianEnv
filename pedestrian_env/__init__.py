from gymnasium.envs.registration import register

register(
    id="PedestrianEnv-v0",
    entry_point="pedestrian_env.envs:PedestrianEnv",
)
