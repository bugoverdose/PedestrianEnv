from dqn import run_DQN_CnnPolicy, test_policy, visualize_test

def best_so_far():
    # evades cars well, not stupid, can't use crosswalks
    model_name = "dqn_3_64_64_fd256_kernel3_1"
    run_DQN_CnnPolicy(total_timesteps=500_000,
            features_dim=256, # 128, 256, 512
            filters_per_group=3,
            n_output_channels=[64, 64],
            kernel_size=3, # 3,5,7
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
    test_policy(model_name) # test score: 1585.5000
    visualize_test(model_name)

def run_default(n_output_channels, features_dim, frame_stack=0, run=True):
    if frame_stack > 0:
        model_name = f"dqn_obs_fs{frame_stack}_{n_output_channels[0]}_{n_output_channels[1]}_{n_output_channels[2]}_fd{features_dim}_kernel3_1"
    else:
        model_name = f"dqn_obs_{n_output_channels[0]}_{n_output_channels[1]}_{n_output_channels[2]}_fd{features_dim}_kernel3_1"
    if run:
        run_DQN_CnnPolicy(total_timesteps=500_000,
            features_dim=features_dim, # 128, 256, 512
            filters_per_group=n_output_channels[0],
            n_output_channels=[n_output_channels[1], n_output_channels[2]],
            kernel_size=3, # 3,5,7
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
    visualize_test(model_name)

if __name__ == "__main__":
    # best_so_far()
    run_default(run=True, n_output_channels = (5, 64, 64), features_dim = 256, frame_stack = 0)
    # features_dim = 128 # [128, 256, 512, 64, 32, 16]
    # n_output_channels = (5, 64, 64)
    # n_output_channels = (3, 64, 64)
    # n_output_channels = (3, 32, 64)
