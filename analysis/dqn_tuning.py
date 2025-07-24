from dqn import run_DQN

if __name__ == "__main__":
    # dqn_episode_1 # does nothing
    # run_DQN(total_timesteps=1_000_000,
    #         net_arch=[256, 256, 256],
    #         learning_rate=1e-4, 
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.8,
    #         exploration_final_eps = 0.1,
    #         train_freq = (4, "episode"),
    #         gradient_steps=1,
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         tb_log_name="dqn_episode") # 0.0000

    run_DQN(total_timesteps=1_000_000,
            net_arch=[512, 512, 512], #
            learning_rate=1e-4, 
            exploration_initial_eps=1.0,
            exploration_fraction = 0.8,
            exploration_final_eps = 0.1,
            train_freq = (4, "episode"),
            gradient_steps=1,
            tau=1.0, # hard
            target_update_interval = 50,
            buffer_size = 10_000,
            batch_size=32,
            learning_starts = 10_000,
            tb_log_name="dqn_episode") # 0.0000

    # =====================================

    # dqn_tuning_1
    # run_DQN(total_timesteps=300_000,
    #         net_arch=[256, 256, 256],
    #         learning_rate=1e-4, 
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.6,
    #         exploration_final_eps = 0.05,
    #         train_freq = 1, # vs (4, "episode")
    #         gradient_steps=1,
    #         tau=1.0, # hard
    #         target_update_interval = 500,
    #         buffer_size = 5_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         tb_log_name="dqn_tuning") # -258.2500