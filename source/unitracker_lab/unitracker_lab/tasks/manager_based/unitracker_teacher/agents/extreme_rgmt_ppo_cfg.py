"""Extreme-RGMT Stage-II PACE/STAR configuration."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import UnitrackerTeacherPPORunnerCfg


@configclass
class ExtremeRGMTPACEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "PACEPPO"
    consolidation_base_weight: float = 0.3
    consolidation_gain: float = 5.0
    acquisition_ratio_reference: float = 0.6
    acquisition_ratio_ema_beta: float = 0.99
    star_enabled: bool = True
    star_topk_fraction: float = 0.05
    star_resample_fraction: float = 0.25


@configclass
class UnitrackerExtremeRGMTPPORunnerCfg(UnitrackerTeacherPPORunnerCfg):
    experiment_name = "unitracker_extreme_rgmt"
    run_name = "stage2_pace_star"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # Must remain checkpoint-compatible with this repository's Stage-I
        # teacher. The paper's larger network belongs to the future Stage-I
        # architecture reproduction, not to the PACE/STAR integration.
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )
    algorithm = ExtremeRGMTPACEAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )
