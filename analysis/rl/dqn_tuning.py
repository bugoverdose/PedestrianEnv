from dqn import run_DQN_CnnPolicy, test_policy, visualize_test

if __name__ == "__main__":
    pass
    # CNNFeaturesExtractor based


    # Evades cars, but sometimes dies because of running into the road even though a car is right in front of it. Don't use crosswalks,
    # n_output_channels1 = 16
    # n_output_channels2 = 32
    # n_output_channels3 = 32
#     model_name = "dqn_cnn_group1_layer3_fd128_kernel3_1"
    # dqn_cnn_group1_layer3_fd128_kernel3_1: test score: 962.0000
    model_name = "dqn_cnn_group1_layer3_fd256_kernel3_1"
    # dqn_cnn_group1_layer3_fd256_kernel3_1: test score: 1068.5000
#     model_name = "dqn_cnn_group1_layer3_fd512_kernel3_1"
    # dqn_cnn_group1_layer3_fd512_kernel3_1: test score: 856.5000

#     model_name = "dqn_cnn_group1_layer3_fd128_kernel5_1"
    # dqn_cnn_group1_layer3_fd128_kernel5_1: test score: 914.5000
#     model_name = "dqn_cnn_group1_layer3_fd256_kernel5_1"
    # dqn_cnn_group1_layer3_fd256_kernel5_1: test score: 515.5000
#     model_name = "dqn_cnn_group1_layer3_fd512_kernel5_1"
    # dqn_cnn_group1_layer3_fd512_kernel5_1: test score: 758.5000

#     model_name = "dqn_cnn_group1_layer3_fd128_kernel7_1"
    # dqn_cnn_group1_layer3_fd128_kernel7_1: test score: 747.0000
#     model_name = "dqn_cnn_group1_layer3_fd512_kernel7_1"
    # dqn_cnn_group1_layer3_fd512_kernel7_1: test score: 621.0000
#     model_name = "dqn_cnn_group1_layer3_fd256_kernel7_1"
    # dqn_cnn_group1_layer3_fd256_kernel7_1: test score: 978.5000
#     run_DQN_CnnPolicy(total_timesteps=500_000,
#             features_dim=512, # 128, 256, 512
#             kernel_size=3, # 3,5,7
#             learning_rate=1e-4,
#             exploration_initial_eps=1.0,
#             exploration_fraction = 0.9,
#             exploration_final_eps = 0.01,
#             train_freq = (4, "episode"),
#             gradient_steps = -1,
#             tau=1.0, # hard
#             target_update_interval = 50,
#             buffer_size = 10_000,
#             batch_size=32,
#             learning_starts = 10_000,
#             saved_model_name=model_name,
#             tb_log_name=model_name[:-2])
#     test_policy(model_name)

    # 2 layers are not enough? bad performance
    # n_output_channels1 = 32
    # n_output_channels2 = 64
#     model_name = "dqn_cnn_group1_layer2_fd256_kernel3_1"
#     run_DQN_CnnPolicy(total_timesteps=500_000,
#             features_dim=256, kernel_size=3,
#             learning_rate=1e-4,
#             exploration_initial_eps=1.0,
#             exploration_fraction = 0.9,
#             exploration_final_eps = 0.01,
#             train_freq = (4, "episode"),
#             gradient_steps = -1,
#             tau=1.0, # hard
#             target_update_interval = 50,
#             buffer_size = 10_000,
#             batch_size=32,
#             learning_starts = 10_000,
#             saved_model_name=model_name,
#             tb_log_name=model_name[:-2])
    # test score: 156.5000 

    # SimpleCNN required ==============================================
    # BEST (group O, 3 layers)
    # highest score, evades cars well, doesn't use crosswalks
#     model_name = "dqn_cnn_fd256_kernel3_5" # binary only crosswalk info
#     run_DQN_CnnPolicy(total_timesteps=500_000,
#             features_dim=256, kernel_size=3,
#             learning_rate=1e-4,
#             exploration_initial_eps=1.0,
#             exploration_fraction = 0.9,
#             exploration_final_eps = 0.01,
#             train_freq = (4, "episode"),
#             gradient_steps = -1,
#             tau=1.0, # hard
#             target_update_interval = 50,
#             buffer_size = 10_000,
#             batch_size=32,
#             learning_starts = 10_000,
#             saved_model_name=model_name,
#             tb_log_name="dqn_cnn_fd256_kernel3")
#     visualize_test(model_name)
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
    # =========================================================
    test_policy(model_name)
    visualize_test(model_name)
