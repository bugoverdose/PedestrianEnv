from ppo import run_PPO_CnnPolicy, test_policy, visualize_test

def best_so_far():
    pass
    # model_name = ""
    # test_policy(model_name)
    # visualize_test(model_name)

def run_default(vf_coef, clip_range, n_steps, n_epochs,
                ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.9, 
                kernel_size=3, features_dim=256, n_output_channels=[3, 64, 64], 
                total_timesteps=1_000_000, run=True):
    model_name = f"ppo_vf_coef{vf_coef}_cr{clip_range}_n_steps{n_steps}_n_epochs{n_epochs}_ts{total_timesteps}_1"
    if run:
        run_PPO_CnnPolicy(features_dim=features_dim,
                        filters_per_group=n_output_channels[0],
                        n_output_channels=n_output_channels[1:],
                        kernel_size=kernel_size,
                        learning_rate=1e-4,
                        n_steps=n_steps, # 100 too small? default=2048
                        batch_size=32,
                        n_epochs=n_epochs,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=clip_range,
                        ent_coef_init=ent_coef_init,
                        ent_coef_final=ent_coef_final,
                        ent_coef_fraction=ent_coef_fraction,
                        vf_coef=vf_coef,
                        max_grad_norm=0.5,
                        total_timesteps=total_timesteps,
                        saved_model_name=model_name,
                        tb_log_name=model_name[:-2])
    test_policy(model_name)
    # visualize_test(model_name)

if __name__ == "__main__":
    # local
    run_default(run=True, vf_coef=0.1, clip_range=0.3, n_steps=2048, n_epochs=2, total_timesteps=1_000_000)
    run_default(run=True, vf_coef=0.1, clip_range=0.3, n_steps=1024, n_epochs=2, total_timesteps=1_000_000)

    # CCSL3 (8/1): nohup python ppo_tuning.py > 08.01_ppo1.log 2>&1 &
    # run_default(run=True, vf_coef=0.1, clip_range=0.3, n_steps=128, n_epochs=2, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.3, n_steps=256, n_epochs=2, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.3, n_steps=512, n_epochs=2, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.3, n_steps=128, n_epochs=3, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.3, n_steps=256, n_epochs=3, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.3, n_steps=512, n_epochs=3, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.4, n_steps=128, n_epochs=2, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.4, n_steps=256, n_epochs=2, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.4, n_steps=512, n_epochs=2, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.4, n_steps=128, n_epochs=3, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.4, n_steps=256, n_epochs=3, total_timesteps=1_000_000)
    # run_default(run=True, vf_coef=0.1, clip_range=0.4, n_steps=512, n_epochs=3, total_timesteps=1_000_000)

    # ========================================================================
    # very stupid, can't evade cars
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.6, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.9, kernel_size=3, features_dim=256, n_output_channels=[3, 64, 64])
    # ppo_fixed_cr0.6_ent_coef_frac0.9_ch3_64_64_fd256_kernel3_1: test score: -196.5000
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.5, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.9, kernel_size=3, features_dim=256, n_output_channels=[3, 64, 64])
    # ppo_fixed_cr0.5_ent_coef_frac0.9_ch3_64_64_fd256_kernel3_1: test score: -22.5000
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.4, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.9, kernel_size=3, features_dim=256, n_output_channels=[3, 64, 64])
    # ppo_fixed_cr0.4_ent_coef_frac0.9_ch3_64_64_fd256_kernel3_1: test score: -45.0000
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.6, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.8, kernel_size=3, features_dim=256, n_output_channels=[3, 64, 64])
    # ppo_fixed_cr0.6_ent_coef_frac0.8_ch3_64_64_fd256_kernel3_1: test score: -29.0000
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.5, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.8, kernel_size=3, features_dim=256, n_output_channels=[3, 64, 64])
    # ppo_fixed_cr0.5_ent_coef_frac0.8_ch3_64_64_fd256_kernel3_1: test score: -108.0000
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.4, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.8, kernel_size=3, features_dim=256, n_output_channels=[3, 64, 64])
    # ppo_fixed_cr0.4_ent_coef_frac0.8_ch3_64_64_fd256_kernel3_1: test score: -45.0000
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.5, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.9, kernel_size=3, features_dim=256, n_output_channels=[5, 64, 64])
    # ppo_fixed_cr0.5_ent_coef_frac0.9_ch5_64_64_fd256_kernel3_1: test score: -134.5000
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.5, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.8, kernel_size=3, features_dim=256, n_output_channels=[5, 64, 64])
    # ppo_fixed_cr0.5_ent_coef_frac0.8_ch5_64_64_fd256_kernel3_1: test score: -29.0000
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.5, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.9, kernel_size=3, features_dim=128, n_output_channels=[3, 64, 64])
    # ppo_fixed_cr0.5_ent_coef_frac0.9_ch3_64_64_fd128_kernel3_1: test score: -6.0000
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.5, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.8, kernel_size=3, features_dim=128, n_output_channels=[3, 64, 64])
    # ppo_fixed_cr0.5_ent_coef_frac0.8_ch3_64_64_fd128_kernel3_1: test score: -33.5000
    # run_default_old(run=False, total_timesteps=500_000, clip_range=0.5, ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.9, kernel_size=3, features_dim=128, n_output_channels=[5, 64, 64])
    # ppo_fixed_cr0.5_ent_coef_frac0.9_ch5_64_64_fd128_kernel3_1: test score: -25.0000
    # ==============================

def run_default_old(total_timesteps, clip_range, ent_coef_init, ent_coef_final, ent_coef_fraction, kernel_size=3, features_dim=128, n_output_channels=[5, 64, 64], run=True):
    model_name = f"ppo_fixed_cr{clip_range}_ent_coef_frac{ent_coef_fraction}_ch{n_output_channels[0]}_{n_output_channels[1]}_{n_output_channels[2]}_fd{features_dim}_kernel{kernel_size}_1"
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
                        clip_range=clip_range,
                        ent_coef_init=ent_coef_init,
                        ent_coef_final=ent_coef_final,
                        ent_coef_fraction=ent_coef_fraction,
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        total_timesteps=total_timesteps,
                        saved_model_name=model_name,
                        tb_log_name=model_name[:-2])
    test_policy(model_name)