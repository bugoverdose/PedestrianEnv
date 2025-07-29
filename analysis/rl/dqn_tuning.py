from dqn import run_DQN_CnnPolicy, visualize_test

if __name__ == "__main__":
    pass
    # BEST for now (SimpleCNN required)
    # highest score, evades cars well, doesn't use crosswalks
    model_name = "dqn_cnn_fd256_kernel3_5" # binary only crosswalk info
    run_DQN_CnnPolicy(total_timesteps=500_000,
            features_dim=256, kernel_size=3,
            learning_rate=1e-4,
            exploration_initial_eps=1.0,
            exploration_fraction = 0.9,
            exploration_final_eps = 0.01,
            train_freq = (4, "episode"),
            gradient_steps = -1,
            tau=1.0, # hard
            target_update_interval = 50,
            buffer_size = 10_000,
            batch_size=32,
            learning_starts = 10_000,
            saved_model_name=model_name,
            tb_log_name="dqn_cnn_fd256_kernel3")
    visualize_test(model_name)
    # test score: 1077.0000

    # model_name = "dqn_cnn_fd256_kernel3_3" # best learning graph. mean rewards reaching over 900
    # run_DQN_CnnPolicy(total_timesteps=500_000, # better than 300_000, 1_000_000
    #         features_dim=256, kernel_size=3,
    #         learning_rate=1e-4,
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.9, #
    #         exploration_final_eps = 0.01, #
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1,
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_fd256_kernel3")
    # test score: 949.0000
    # visualize_test(model_name)

    # good learning graph, but only move east to evade cars & can't use crosswalks
    # model_name = "dqn_cnn_fd256_kernel3_4" # more crosswalk closeness info
    # run_DQN_CnnPolicy(total_timesteps=500_000,
    #         features_dim=256, kernel_size=3,
    #         learning_rate=1e-4,
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.9,
    #         exploration_final_eps = 0.01,
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1,
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_fd256_kernel3")
    # visualize_test(model_name)
    # test score: 979.5000

    # NOTE: No reward shaping, but bonus on reaching the end of the map
    # BEST for now: Evades cars, but didn't learn to use crosswalks
    # saved model: dqn_cnn_group_fd256_gs_auto_1.zip
    # log: dqn_cnn_group_fd256_gs_auto_1
    # model_name = "dqn_cnn_group_fd256_gs_auto_1" 
    # run_DQN_CnnPolicy(total_timesteps=1_000_000,
    #         learning_rate=1e-4, 
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.6,
    #         exploration_final_eps = 0.05,
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1, # 1
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_group_fd512_gs_auto")
    # test score: 355.0000

    # model_name = "dqn_cnn_fd256_kernel3_1" # similar to "dqn_cnn_group_fd256_gs_auto_1"
    # run_DQN_CnnPolicy(total_timesteps=1_000_000,
    #         features_dim=256, kernel_size=3,
    #         learning_rate=1e-4,
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.6,
    #         exploration_final_eps = 0.05,
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1, # 1
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_fd256_kernel3")

    # saved model: dqn_cnn_fd256_kernel5_padding1.zip
    # log: dqn_cnn_group_fd512_kernel5_1
    # sometimes moves in a weird way, but can evade cars and cross roads
    # issue: no bonus on reaching the end of the map

    # BAD: goes back to the crossed road and dies
    # model_name = "dqn_cnn_fd256_kernel3_2"
    # run_DQN_CnnPolicy(total_timesteps=300_000,
    #         features_dim=256, kernel_size=3,
    #         learning_rate=1e-4,
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.9,
    #         exploration_final_eps = 0.01,
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1,
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_fd256_kernel3")
    # test score: 282.0000

    # =============================================

    # model_name = "dqn_cnn_fd512_kernel3_1"
    # run_DQN_CnnPolicy(total_timesteps=1_000_000,
    #         features_dim=512, kernel_size=3,
    #         learning_rate=1e-4,
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.8,
    #         exploration_final_eps = 0.1,
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1, # 1
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_fd512_kernel3")
    # visualize_test(model_name)

    # model_name = "dqn_cnn_fd512_kernel5_1"
    # run_DQN_CnnPolicy(total_timesteps=1_000_000,
    #         features_dim=512, kernel_size=5,
    #         learning_rate=1e-4,
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.8,
    #         exploration_final_eps = 0.1,
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1, # 1
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_fd512_kernel5")
    # visualize_test(model_name)

    # model_name = "dqn_cnn_fd512_kernel7_1"
    # run_DQN_CnnPolicy(total_timesteps=1_000_000,
    #         features_dim=512, kernel_size=7,
    #         learning_rate=1e-4,
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.8,
    #         exploration_final_eps = 0.1,
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1, # 1
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_fd512_kernel7")
    # visualize_test(model_name)
