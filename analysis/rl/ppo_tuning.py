from ppo import run_PPO_CnnPolicy, visualize_test

if __name__ == "__main__":
    model_name = "ppo_cnn_fd256_kernel3_ent_coef0.1_3"
    run_PPO_CnnPolicy(features_dim=256, kernel_size=3,
                      learning_rate=1e-4,
                      n_steps=2048,
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

    # =================================================
    # groups=1
    # model_name = "ppo_cnn_fd256_kernel3_ent_coef0.1_3"
    # run_PPO_CnnPolicy(features_dim=256, kernel_size=3,
    #                   learning_rate=1e-4,
    #                   n_steps=2048,
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
    # test score: 0.0000

    # ent_coef=0.01
    # model_name = "ppo_cnn_fd256_kernel3_ent_coef0.1_2"
    # run_PPO_CnnPolicy(features_dim=256, kernel_size=3,
    #                   learning_rate=1e-4,
    #                   n_steps=2048,
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
    # test score: 0.0000

    # ent_coef=0.05
    # model_name = "ppo_cnn_fd256_kernel3_ent_coef0.05_2" # go WEST only
    # run_PPO_CnnPolicy(features_dim=256, kernel_size=3,
    #                   learning_rate=1e-4,
    #                   n_steps=2048,
    #                   batch_size=64,
    #                   n_epochs=10,
    #                   gamma=0.99,
    #                   gae_lambda=0.95,
    #                   clip_range=0.2,
    #                   ent_coef=0.05,
    #                   vf_coef=0.5,
    #                   max_grad_norm=0.5,
    #                   total_timesteps=300_000,
    #                   saved_model_name=model_name,
    #                   tb_log_name=model_name[:-2])
    # test score: 0.0000

    # ent_coef=0.01
    # reward curve converges under 0
    # model_name = "ppo_cnn_fd256_kernel3_fast_2" # stopped training, but will stay away from the road
    # run_PPO_CnnPolicy(features_dim=256, kernel_size=3,
    #                   learning_rate=1e-4,
    #                   n_steps=2048,
    #                   batch_size=64,
    #                   n_epochs=10,
    #                   gamma=0.99,
    #                   gae_lambda=0.95,
    #                   clip_range=0.2,
    #                   ent_coef=0.01,
    #                   vf_coef=0.5,
    #                   max_grad_norm=0.5,
    #                   total_timesteps=300_000,
    #                   saved_model_name=model_name,
    #                   tb_log_name=model_name[:-2])

    # model_name = "ppo_cnn_fd256_kernel3_fast_1" # fast to learn, but only goes DOWN
    # run_PPO_CnnPolicy(features_dim=256, kernel_size=3,
    #                   learning_rate=1e-4,
    #                   n_steps=2048,
    #                   batch_size=64,
    #                   n_epochs=10,
    #                   gamma=0.99,
    #                   gae_lambda=0.95,
    #                   clip_range=0.2,
    #                   ent_coef=0.01,
    #                   vf_coef=0.5,
    #                   max_grad_norm=0.5,
    #                   total_timesteps=100_000,
    #                   saved_model_name=model_name,
    #                   tb_log_name=model_name[:-2])
    # test score: 0.0000

    # =================================================
    visualize_test(model_name)
