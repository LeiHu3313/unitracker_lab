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


def _policy_cfg() -> RslRlPpoActorCriticCfg:
    return RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # Interim Stage-I proxy. The paper encoder will replace this MLP in a
        # separate milestone without coupling that work to PACE/STAR.
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
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
