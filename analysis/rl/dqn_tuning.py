from dqn import run_DQN_CnnPolicy, test_policy, visualize_test

def best_so_far():
    # evades cars well, goes the other direction of the car in front of it.
    model_name = "dqn_obs_3_64_64_fd256_kernel3_1"
    features_dim = 256
    n_output_channels = (3, 64, 64)
    kernel_size=3
    frame_stack=0
    run_DQN_CnnPolicy(total_timesteps=500_000,
        features_dim=features_dim, # 128, 256, 512
        filters_per_group=n_output_channels[0],
        n_output_channels=[n_output_channels[1], n_output_channels[2]],
        kernel_size=kernel_size, # 3,5,7
        frame_stack=frame_stack,
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
        tb_log_name=model_name[:-2])
    test_policy(model_name) # test score: 1648.5000
    visualize_test(model_name)

def run_default(n_output_channels, features_dim, kernel_size=3, frame_stack=0, run=True):
    model_name = f"dqn_obs_{n_output_channels[0]}_{n_output_channels[1]}_{n_output_channels[2]}_fd{features_dim}_kernel{kernel_size}_1"
    if run:
        run_DQN_CnnPolicy(total_timesteps=500_000,
            features_dim=features_dim, # 128, 256, 512
            filters_per_group=n_output_channels[0],
            n_output_channels=[n_output_channels[1], n_output_channels[2]],
            kernel_size=kernel_size, # 3,5,7
            frame_stack=frame_stack,
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
            tb_log_name=model_name[:-2])
    test_policy(model_name)
    # visualize_test(model_name)

if __name__ == "__main__":
    best_so_far()
    # run_default(run=False, n_output_channels = (3, 64, 64), features_dim = 256, kernel_size=3)
    # dqn_obs_3_64_64_fd256_kernel3_1: test score: 1648.5000
    # best learning curve: 1,632.177

    # run_default(run=False, n_output_channels = (5, 64, 64), features_dim = 256, kernel_size=3)
    # dqn_obs_5_64_64_fd256_kernel3_1: test score: 1622.5000
    # run_default(run=False, n_output_channels = (3, 32, 64), features_dim = 128, kernel_size=3)
    # dqn_obs_3_32_64_fd64_kernel3_1: test score: 1617.0000
    # run_default(run=False, n_output_channels = (5, 64, 64), features_dim = 64, kernel_size=5)
    # dqn_obs_3_32_64_fd128_kernel3_1: test score: 1604.0000
    # learning curve: 1,559.7673

    # run_default(run=False, n_output_channels = (3, 64, 64), features_dim = 128, kernel_size=3)
    # dqn_obs_3_64_64_fd128_kernel3_1: test score: 1536.0000
    # run_default(run=False, n_output_channels = (5, 64, 64), features_dim = 128, kernel_size=3)
    # dqn_obs_5_64_64_fd128_kernel3_1: test score: 1380.0000
    # run_default(run=False, n_output_channels = (5, 64, 64), features_dim = 64, kernel_size=3)
    # dqn_obs_5_64_64_fd64_kernel3_1: test score: 1289.0000
    # run_default(run=False, n_output_channels = (3, 32, 64), features_dim = 256, kernel_size=3)
    # dqn_obs_3_32_64_fd256_kernel3_1: test score: 1257.0000
    # run_default(run=False, n_output_channels = (3, 32, 64), features_dim = 64, kernel_size=3)
    # dqn_obs_5_64_64_fd64_kernel5_1: test score: 1289.0000
    # run_default(run=False, n_output_channels = (3, 32, 64), features_dim = 256, kernel_size=5)
    # dqn_obs_3_32_64_fd256_kernel5_1: test score: 1257.0000

    # TODO: CCSL3 (8/1): nohup python dqn_tuning.py > 08.01_full.log 2>&1 &
    # run_default(run=False, n_output_channels = (3, 32, 64), features_dim = 128, kernel_size=5)
    # run_default(run=False, n_output_channels = (3, 32, 64), features_dim = 64, kernel_size=5)
