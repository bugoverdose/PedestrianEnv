from dqn import run_DQN_CnnPolicy, visualize_test

if __name__ == "__main__":
    model_name = "dqn_cnn_group_fd512_gs_auto_2"
    run_DQN_CnnPolicy(total_timesteps=1_000_000,
            learning_rate=1e-4, 
            exploration_initial_eps=1.0,
            exploration_fraction = 0.6,
            exploration_final_eps = 0.05,
            train_freq = (4, "episode"),
            gradient_steps = -1, # 1
            tau=1.0, # hard
            target_update_interval = 50,
            buffer_size = 10_000,
            batch_size=32,
            learning_starts = 10_000,
            saved_model_name=model_name,
            tb_log_name="dqn_cnn_group_fd512_gs_auto")
    visualize_test(model_name)

    # kernel_size=3, 3 layers of Conv2d
    # model_name = "dqn_cnn_group_fd512_gs_auto_1"
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
    # visualize_test(model_name)
    
    # model_name = "dqn_cnn_group_fd512_kernel7_1"
    # run_DQN_CnnPolicy(total_timesteps=500_000,
    #         learning_rate=1e-4,
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.8,
    #         exploration_final_eps = 0.05,
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1, # 1
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_group_fd512_kernel7")
    # visualize_test(model_name)

    # model_name = "dqn_cnn_crosswalk_reward_1" # "dqn_cnn_group_fd512_gs_auto_1"
    # run_DQN_CnnPolicy(total_timesteps=700_000,
    #         learning_rate=1e-4, 
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.9,
    #         exploration_final_eps = 0.15,
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1, # 1
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_crosswalk_reward")
    # visualize_test(model_name)

    # moves up,left,right,down to reach almost the end. 
    # Problem: Doesn't reach the end of the road!!!
    # model_name = "dqn_cnn_group_fd512_kernel5_1"
    # run_DQN_CnnPolicy(total_timesteps=1_000_000,
    #         learning_rate=1e-4,
    #         exploration_initial_eps=1.0,
    #         exploration_fraction = 0.8,
    #         exploration_final_eps = 0.05,
    #         train_freq = (4, "episode"),
    #         gradient_steps = -1, # 1
    #         tau=1.0, # hard
    #         target_update_interval = 50,
    #         buffer_size = 10_000,
    #         batch_size=32,
    #         learning_starts = 10_000,
    #         saved_model_name=model_name,
    #         tb_log_name="dqn_cnn_group_fd512_kernel5")
    # visualize_test(model_name)