"""PPO baseline for the 1425-D fully privileged MimicLite-style G1 teacher."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class MimicLiteAlignedPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """RSL-RL PPO fields needed for MimicLite's linear entropy schedule."""

    entropy_coef_start: float = 0.012
    entropy_coef_end: float = 0.001
    entropy_decay_start: int = 3250
    entropy_decay_end: int = 3500


@configclass
class UnitrackerTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 1_000
    experiment_name = "unitracker_teacher"
    run_name = "g1_stage1"
    clip_actions = None
    # Deliberately identical raw inputs: this is a fully privileged teacher.
    # RSL-RL keeps independent actor/critic normalizers, which is desirable.
    obs_groups = {
        "policy": ["teacher_state", "teacher_reference"],
        "critic": ["teacher_state", "teacher_reference"],
    }
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # Keep the teacher network ABI so existing checkpoints remain loadable.
        # MimicLite's deployable actor has different inputs and is therefore not
        # an apples-to-apples architecture target for this privileged teacher.
        actor_hidden_dims=[1024, 512, 512, 256],
        critic_hidden_dims=[1024, 512, 512, 256],
        activation="elu",
    )
    algorithm = MimicLiteAlignedPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Initial value retained for compatibility with callers that do not
        # invoke PPO.set_iteration; OnPolicyRunner applies the schedule below.
        entropy_coef=0.012,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )
