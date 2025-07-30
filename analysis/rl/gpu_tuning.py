from dqn import run_DQN_CnnPolicy, test_policy, visualize_test

def grid_search():
    features_dims = [8, 16, 32, 64, 128, 256, 512]
    n_output_channels_list = [(5, 64, 64), (3, 64, 64), (3, 32, 64)]
    for features_dim in features_dims:
        for n_output_channels in n_output_channels_list:
            model_name = f"dqn_obs_{n_output_channels[0]}_{n_output_channels[1]}_{n_output_channels[2]}_fd{features_dim}_kernel3_1"
            run_DQN_CnnPolicy(total_timesteps=500_000,
                    features_dim=features_dim, # 128, 256, 512
                    filters_per_group=n_output_channels[0],
                    n_output_channels=[n_output_channels[1], n_output_channels[2]],
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

if __name__ == "__main__":
    grid_search() # CCSL3에서 실행 중
