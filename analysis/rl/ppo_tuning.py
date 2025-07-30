from ppo import run_PPO_CnnPolicy, visualize_test

if __name__ == "__main__":
    # model_name = "ppo_3_64_64_fd256_kernel3_1" # n_steps=200 # do nothing
    # model_name = "ppo_3_64_64_fd256_kernel3_2" # n_steps=100 # ?
    # run_PPO_CnnPolicy(features_dim=256, 
    #                   filters_per_group=3,
    #                   n_output_channels=[64, 64],
    #                   kernel_size=3,
    #                   learning_rate=1e-4,
    #                   n_steps=100,
    #                   batch_size=64,
    #                   n_epochs=10,
    #                   gamma=0.99,
    #                   gae_lambda=0.95,
    #                   clip_range=0.2,
    #                   ent_coef=0.1,
    #                   vf_coef=0.5,
    #                   max_grad_norm=0.5,
    #                   total_timesteps=300_000,
    #                   saved_model_name=model_name,
    #                   tb_log_name=model_name[:-2])

    model_name = "ppo_5_64_64_fd256_kernel3_2"
    run_PPO_CnnPolicy(features_dim=256, 
                      filters_per_group=5,
                      n_output_channels=[64, 64],
                      kernel_size=3,
                      learning_rate=1e-4,
                      n_steps=100,
                      batch_size=64,
                      n_epochs=10,
                      gamma=0.99,
                      gae_lambda=0.95,
                      clip_range=0.2,
                      ent_coef=0.1,
                      vf_coef=0.5,
                      max_grad_norm=0.5,
                      total_timesteps=300_000,
                      saved_model_name=model_name,
                      tb_log_name=model_name[:-2])
