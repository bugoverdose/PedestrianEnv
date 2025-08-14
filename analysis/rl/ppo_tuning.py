from ppo import run_PPO_CnnPolicy, test_policy, visualize_test
from types import FunctionType

def best_so_far():
    pass
    # model_name = ""
    # test_policy(model_name)
    # visualize_test(model_name)

def run_default(vf_coef, clip_range, n_steps, n_epochs, fixed_episode_seed_range, gae_lambda, max_grad_norm, batch_size=32,
                ent_coef_init=1.0, ent_coef_final=0.01, ent_coef_fraction=0.9, 
                kernel_size=3, features_dim=128, n_output_channels=[5, 64, 64], 
                total_timesteps=1_000_000, run=True, trial=1):
    
    # if fixed_episode_seed_range is None:
    #     model_name = f"ppo_no_ep_seed"
    # elif fixed_episode_seed_range[0] == fixed_episode_seed_range[1]:
    #     model_name = f"ppo_ep_seed{fixed_episode_seed_range[0]}"
    # else:
    #     model_name = f"ppo_ep_seed{fixed_episode_seed_range[0]}_{fixed_episode_seed_range[1]}"
    # model_name = model_name + f"_ent_coef{ent_coef_init}_{ent_coef_final}_{ent_coef_fraction}_vf_coef{vf_coef}_n_steps{n_steps}_n_epochs{n_epochs}_batch_size{batch_size}_ts{total_timesteps}_{trial}"
    model_name = f"ppo_gae{gae_lambda}_vf_coef{vf_coef}_max_grad_norm{max_grad_norm}_n_steps{n_steps}_n_epochs{n_epochs}_batch_size{batch_size}_ent_coef{ent_coef_init}_{ent_coef_final}_{ent_coef_fraction}_ts{total_timesteps}"
    if not isinstance(clip_range, FunctionType):
        model_name = model_name + f"_cr{clip_range}"
    model_name = model_name + f"_{trial}"
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
                        gae_lambda=gae_lambda,
                        clip_range=clip_range,
                        ent_coef_init=ent_coef_init,
                        ent_coef_final=ent_coef_final,
                        ent_coef_fraction=ent_coef_fraction,
                        vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm,
                        fixed_episode_seed_range=fixed_episode_seed_range,
                        total_timesteps=total_timesteps,
                        saved_model_name=model_name,
                        tb_log_name=model_name[:-2])
    test_policy(model_name)
    # visualize_test(model_name)

def linear_schedule(initial_value):
    def func(progress):
        return progress * initial_value
    return func

# ppo_ep_seed1001_ent_coef0.5_0.01_0.6_vf_coef0.3_n_steps4_n_epochs2_batch_size4_ts500000_1 : GOOD
# ppo_ep_seed1001_ent_coef0.5_0.1_0.9_vf_coef0.4_n_steps4_n_epochs2_batch_size4_ts1000000_2 : GOOD?

# ppo_no_ep_seed_ent_coef0.5_0.1_0.9_vf_coef0.4_n_steps4_n_epochs2_batch_size8_ts1000000_1 : ??

# ppo_ep_seed1001_ent_coef0.5_0.01_0.6_vf_coef0.3_n_steps4_n_epochs2_batch_size4_ts500000_1_cr0_1
# ppo_ep_seed1001_ent_coef0.5_0.1_0.8_vf_coef0.2_n_steps4_n_epochs2_batch_size4_ts1000000_1
if __name__ == "__main__":
    # Best candidate
    # run_default(run=False, vf_coef=0.3, clip_range=linear_schedule(1.0), n_steps=4, n_epochs=2, batch_size=4, gae_lambda=0.95, max_grad_norm=0.5, fixed_episode_seed_range=(1001, 1001), \
    #     ent_coef_init=0.5, ent_coef_final=0.01, ent_coef_fraction=0.6, total_timesteps=500_000)
    # ppo_ep_seed1001_ent_coef0.5_0.01_0.6_vf_coef0.3_n_steps4_n_epochs2_batch_size4_ts500000_1: test score: 249.5000
    # run_default(run=False, vf_coef=0.4, clip_range=linear_schedule(1.0), n_steps=4, n_epochs=2, batch_size=4, gae_lambda=0.95, max_grad_norm=0.5, fixed_episode_seed_range=(1001, 1001), \
    #     ent_coef_init=0.5, ent_coef_final=0.1, ent_coef_fraction=0.9, total_timesteps=1_000_000)
    # ppo_ep_seed1001_ent_coef0.5_0.1_0.9_vf_coef0.4_n_steps4_n_epochs2_batch_size4_ts1000000_1: test score: 0.0000

    # Ongoing1
    # gae_lambda_list = [0.90, 0.95]
    # vf_coef_list = [0.3, 0.5, 0.6]
    # clip_range_list = [linear_schedule(1.0), 0.5]
    # n_steps_list = [128]
    # n_epochs_list = [2, 4]
    # batch_size_list = [16, 32, 64]
    # for gae_lambda in gae_lambda_list:
    #     for vf_coef in vf_coef_list:
    #         for clip_range in clip_range_list:
    #             for n_steps in n_steps_list:
    #                 for n_epochs in n_epochs_list:
    #                     for batch_size in batch_size_list:
    #                         run_default(vf_coef=vf_coef, 
    #                                     clip_range=clip_range,
    #                                     n_steps=n_steps, 
    #                                     n_epochs=n_epochs, 
    #                                     batch_size=batch_size, 
    #                                     gae_lambda = gae_lambda,
    #                                     max_grad_norm=1.0,
    #                                     fixed_episode_seed_range=None,
    #                                     ent_coef_init=0.5, ent_coef_final=0.1, ent_coef_fraction=0.9, 
    #                                     total_timesteps=500_000)

    # =====
    # gae_lambda = 0.9
    # vf_coef = 0.3
    # clip_range = 0.5
    # n_steps = 128
    # n_epochs = 4
    # batch_size = 32
    # total_timesteps = 2_000_000 # 500_000, 1_000_000
    # run_default(vf_coef=vf_coef, 
    #             clip_range=clip_range,
    #             n_steps=n_steps, 
    #             n_epochs=n_epochs, 
    #             batch_size=batch_size, 
    #             gae_lambda = gae_lambda,
    #             max_grad_norm=1.0,
    #             fixed_episode_seed_range=None,
    #             ent_coef_init=0.5, ent_coef_final=0.1, ent_coef_fraction=0.9, 
    #             total_timesteps=total_timesteps, run=False)
    # ppo_gae0.9_vf_coef0.3_max_grad_norm1.0_n_steps128_n_epochs4_batch_size16_ent_coef0.5_0.1_0.9_ts500000_cr0.5_1: test score: -496.0000
    # ppo_gae0.9_vf_coef0.3_max_grad_norm1.0_n_steps128_n_epochs4_batch_size16_ent_coef0.5_0.1_0.9_ts1000000_cr0.5_1: test score: 0.0000
    # ppo_gae0.9_vf_coef0.3_max_grad_norm1.0_n_steps128_n_epochs4_batch_size32_ent_coef0.5_0.1_0.9_ts2000000_cr0.5_1: test score: 0.0000

    # ====
    # Ongoing 2
    gae_lambda = 0.9
    clip_range = 0.5
    n_steps = 128
    n_epochs = 4
    total_timesteps = 2_000_000
    batch_size_list = [32, 64]
    vf_coef_list = [0.3, 0.5, 0.7]
    for batch_size in batch_size_list:
        for vf_coef in vf_coef_list:
            if vf_coef == 0.3 and batch_size == 32: continue
            run_default(vf_coef=vf_coef, 
                        clip_range=clip_range,
                        n_steps=n_steps, 
                        n_epochs=n_epochs, 
                        batch_size=batch_size, 
                        gae_lambda = gae_lambda,
                        max_grad_norm=1.0,
                        fixed_episode_seed_range=None,
                        ent_coef_init=0.5, ent_coef_final=0.1, ent_coef_fraction=0.9, 
                        total_timesteps=total_timesteps, run=True)
