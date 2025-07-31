from ppo import run_PPO_CnnPolicy, test_policy, visualize_test

def run_default(total_timesteps, ent_coef_init, ent_coef_final, ent_coef_fraction, kernel_size=3, features_dim=128, n_output_channels=[5, 64, 64], run=True):
    model_name = f"ppo_ent_coef_lin{ent_coef_init}_{ent_coef_final}_{ent_coef_fraction}_ch{n_output_channels[0]}_{n_output_channels[1]}_{n_output_channels[2]}_fd{features_dim}_kernel{kernel_size}_1"
    if run:
        run_PPO_CnnPolicy(features_dim=features_dim,
                        filters_per_group=n_output_channels[0],
                        n_output_channels=n_output_channels[1:],
                        kernel_size=kernel_size,
                        learning_rate=1e-4,
                        n_steps=2048, # 100 too small
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        ent_coef_init=ent_coef_init,
                        ent_coef_final=ent_coef_final,
                        ent_coef_fraction=ent_coef_fraction,
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        total_timesteps=total_timesteps,
                        saved_model_name=model_name,
                        tb_log_name=model_name[:-2])
    test_policy(model_name)
    visualize_test(model_name)

if __name__ == "__main__":
    # run_default(total_timesteps=500_000, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.9, kernel_size=3, features_dim=128, n_output_channels=[5, 64, 64], run=True)
    run_default(total_timesteps=500_000, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.9, kernel_size=3, features_dim=256, n_output_channels=[5, 64, 64], run=True)
