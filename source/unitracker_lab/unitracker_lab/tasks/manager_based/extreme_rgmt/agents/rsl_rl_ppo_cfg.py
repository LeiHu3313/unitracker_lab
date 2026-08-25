"""RSL-RL configuration for the standalone Extreme-RGMT task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


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
class ExtremeRGMTActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "ExtremeRGMTActorCritic"
    proprioception_dim: int = 64
    reference_token_dim: int = 38
    history_length: int = 10
    reference_window_length: int = 21
    state_encoder_dims: list[int] = [128, 64]
    action_encoder_dims: list[int] = [64, 64]
    command_encoder_dims: list[int] = [128, 64]
    # The paper does not publish these three values. They are explicit
    # reproduction choices and are saved in every resolved agent config.
    history_num_layers: int = 2
    attention_num_heads: int = 4
    fsq_num_tokens: int = 2
    fsq_token_dim: int = 32
    fsq_levels: int = 8


def _policy_cfg() -> ExtremeRGMTActorCriticCfg:
    return ExtremeRGMTActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 512],
        activation="elu",
    )


def _ppo_cfg() -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
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


@configclass
class ExtremeRGMTBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50_000
    save_interval = 500
    experiment_name = "extreme_rgmt_base"
    run_name = "stage1_base"
    clip_actions = None
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = _policy_cfg()
    algorithm = _ppo_cfg()


@configclass
class ExtremeRGMTExpansionPPORunnerCfg(ExtremeRGMTBasePPORunnerCfg):
    experiment_name = "extreme_rgmt_expansion"
    run_name = "stage2_pace_star"
    policy = _policy_cfg()
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
