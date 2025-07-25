from dqn import run_DQN_CnnPolicy, visualize_test

if __name__ == "__main__":
    # always go front
    model_name = "dqn_cnn_4"
    run_DQN_CnnPolicy(total_timesteps=1_000_000,
            learning_rate=1e-4, 
            exploration_initial_eps=1.0,
            exploration_fraction = 0.9,
            exploration_final_eps = 0.1,
            train_freq = (4, "episode"),
            gradient_steps=1,
            tau=1.0, # hard
            target_update_interval = 50,
            buffer_size = 10_000,
            batch_size=32,
            learning_starts = 10_000,
            saved_model_name=model_name,
            tb_log_name="dqn_cnn")
    visualize_test(model_name)
    # test score: -401.5000

#     # always go front
#     model_name = "dqn_cnn_3"
#     run_DQN_CnnPolicy(total_timesteps=1_000_000,
#             learning_rate=1e-4, 
#             exploration_initial_eps=1.0,
#             exploration_fraction = 0.9,
#             exploration_final_eps = 0.1,
#             train_freq = (4, "episode"),
#             gradient_steps=1,
#             tau=1.0, # hard
#             target_update_interval = 50,
#             buffer_size = 10_000,
#             batch_size=32,
#             learning_starts = 10_000,
#             saved_model_name=model_name,
#             tb_log_name="dqn_cnn")
#     visualize_test(model_name)
#     # test score: -351.50

#     # does nothing
#     model_name = "dqn_cnn_1" 
#     run_DQN_CnnPolicy(total_timesteps=200_000, # 1_000_000
#             learning_rate=1e-4, 
#             exploration_initial_eps=1.0,
#             exploration_fraction = 0.6,
#             exploration_final_eps = 0.05,
#             train_freq = (4, "episode"),
#             gradient_steps=1,
#             tau=1.0, # hard
#             target_update_interval = 50,
#             buffer_size = 10_000,
#             batch_size=32,
#             learning_starts = 10_000,
#             saved_model_name=model_name,
#             tb_log_name="dqn_cnn")
#     visualize_test(model_name)
#     # test score: 0.0000

#    # always go front
#     model_name = "dqn_cnn_2"
#     run_DQN_CnnPolicy(total_timesteps=1_000_000,
#             learning_rate=1e-4, 
#             exploration_initial_eps=1.0,
#             exploration_fraction = 0.6,
#             exploration_final_eps = 0.05,
#             train_freq = (4, "episode"),
#             gradient_steps=1,
#             tau=1.0, # hard
#             target_update_interval = 50,
#             buffer_size = 10_000,
#             batch_size=32,
#             learning_starts = 10_000,
#             saved_model_name=model_name,
#             tb_log_name="dqn_cnn")
#     visualize_test(model_name)
#     # test score: -418.0000
