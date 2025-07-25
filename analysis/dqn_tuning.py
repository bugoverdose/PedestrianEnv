from dqn import run_DQN_CnnPolicy, visualize_test

if __name__ == "__main__":
    # dqn_cnn_default_1
    # run_DQN_CnnPolicy(total_timesteps=1_000_000,
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
    #         tb_log_name="dqn_cnn_default")
    # test score: -374.7000

    # dqn_cnn_1
    run_DQN_CnnPolicy(total_timesteps=200_000,
            learning_rate=1e-4, 
            exploration_initial_eps=1.0,
            exploration_fraction = 0.6,
            exploration_final_eps = 0.05,
            train_freq = (4, "episode"),
            gradient_steps=1,
            tau=1.0, # hard
            target_update_interval = 50,
            buffer_size = 10_000,
            batch_size=32,
            learning_starts = 10_000,
            saved_model_name="dqn_cnn",
            tb_log_name="dqn_cnn")
    visualize_test("dqn_cnn")

    # dqn_cnn_2
    # run_DQN_CnnPolicy(total_timesteps=200_000,
    #         learning_rate=1e-4, 
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.7,
    #         exploration_final_eps = 0.1,
    #         train_freq = (4, "episode"),
    #         gradient_steps=1,
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name="dqn_cnn_2",
    #         tb_log_name="dqn_cnn")
    # visualize_test("dqn_cnn_2")
