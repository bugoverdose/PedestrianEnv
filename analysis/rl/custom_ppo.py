import sys
import time
from typing import Any, Dict, Optional, Type, Union

import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import SelfOnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean

class CustomPPO(PPO):
    """
    ent_coef_init : initial ent_coef value 
    ent_coef_final : final ent_coef_final
    ent_coef_fraction : linear decay from ent_coef_init to ent_coef_final for (ent_coef_fraction * total steps) steps,
                        ent_coef_fraction should be 0 if no linear decay
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef_init: float = 0.0,
        ent_coef_final: float = 0.0,
        ent_coef_fraction: float = 1.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy = policy,
            env = env,
            learning_rate = learning_rate,
            n_steps = n_steps,
            batch_size = batch_size,
            n_epochs = n_epochs,
            gamma = gamma,
            gae_lambda = gae_lambda,
            clip_range = clip_range,
            clip_range_vf = clip_range_vf,
            normalize_advantage = normalize_advantage,
            ent_coef = ent_coef_init,
            vf_coef = vf_coef,
            max_grad_norm = max_grad_norm,
            use_sde = use_sde,
            sde_sample_freq = sde_sample_freq,
            rollout_buffer_class = rollout_buffer_class,
            rollout_buffer_kwargs = rollout_buffer_kwargs,
            target_kl = target_kl,
            stats_window_size = stats_window_size,
            tensorboard_log = tensorboard_log,
            policy_kwargs = policy_kwargs,
            verbose = verbose,
            seed = seed,
            device =device,
            _init_setup_model = _init_setup_model,
        )
        if ent_coef_init < ent_coef_final:
            raise Exception(f"ent_coef_init({ent_coef_init}) should be bigger than ent_coef_final({ent_coef_final})")
        self.ent_coef_init = ent_coef_init
        self.ent_coef_final = ent_coef_final
        self.ent_coef_fraction = ent_coef_fraction

    # @override
    def train(self) -> None:
        super().train()
        self.logger.record("train/ent_coef", self.ent_coef, exclude="tensorboard")

    # @override
    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            progress = 1 - self._current_progress_remaining
            if progress < self.ent_coef_fraction:
                self.ent_coef = self.ent_coef_init - (progress * (self.ent_coef_init - self.ent_coef_final))

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self
