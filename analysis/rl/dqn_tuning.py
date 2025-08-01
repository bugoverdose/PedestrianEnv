from dqn import run_DQN_CnnPolicy, test_policy, visualize_test

def best_so_far():
    model_name = "dqn_5_64_64_fd128_kernel3_1"
    n_output_channels = (5, 64, 64)
    features_dim = 128
    kernel_size=3
    trial=1
    # run_DQN_CnnPolicy(total_timesteps=500_000,
    #     features_dim=features_dim, # 128, 256, 512
    #     filters_per_group=n_output_channels[0],
    #     n_output_channels=[n_output_channels[1], n_output_channels[2]],
    #     kernel_size=kernel_size, # 3,5,7
    #     frame_stack=0,
    #     learning_rate=1e-4,
    #     exploration_initial_eps=1.0,
    #     exploration_fraction = 0.9,
    #     exploration_final_eps = 0.01,
    #     train_freq = (4, "episode"),
    #     gradient_steps = -1,
    #     tau=1.0, # hard
    #     target_update_interval = 50,
    #     buffer_size = 10_000,
    #     batch_size=32,
    #     learning_starts = 10_000,
    #     saved_model_name=model_name,
    #     tb_log_name=model_name[:-2])
    test_policy(model_name)
    # dqn_5_64_64_fd128_kernel3_1: test score: 1652.5000
    visualize_test(model_name)
    # 6_best_dqn.mov

def run_default(n_output_channels, features_dim, kernel_size=3, frame_stack=0, run=True, trial=1):
    model_name = f"dqn_{n_output_channels[0]}_{n_output_channels[1]}_{n_output_channels[2]}_fd{features_dim}_kernel{kernel_size}_{trial}"
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

    # high score, but can't make use of safe zone, sometimes goes up when a car is approaching fast and gets hit
    # run_default(run=False, n_output_channels = (5, 64, 64), features_dim = 256, kernel_size=3, trial=1)
    # dqn_5_64_64_fd256_kernel3_1: test score: 1613.0000

    # run_default(run=False, n_output_channels = (5, 64, 64), features_dim = 512, kernel_size=3, trial=1)
    # dqn_5_64_64_fd512_kernel3_1: test score: 1585.0000

    # (4, 64, 64): unstable?
    # run_default(run=False, n_output_channels = (4, 64, 64), features_dim = 128, kernel_size=3, trial=1)
    # dqn_4_64_64_fd128_kernel3_1: test score: 1330.0000
    # run_default(run=False, n_output_channels = (4, 64, 64), features_dim = 256, kernel_size=3, trial=1)
    # dqn_4_64_64_fd256_kernel3_1: test score: 1563.0000
    # run_default(run=False, n_output_channels = (4, 64, 64), features_dim = 512, kernel_size=3, trial=1)
    # dqn_4_64_64_fd512_kernel3_1: test score: 982.0000

    # (3, 64, 64): BAD??
    # run_default(run=False, n_output_channels = (3, 64, 64), features_dim = 256, kernel_size=3, trial=1)
    # dqn_3_64_64_fd256_kernel3_1: test score: 160.5000
    # dqn_3_64_64_fd256_kernel3_1: test score: 731.0000
