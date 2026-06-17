from adaptive_quant.configuration import FrameworkConfig

CONFIG = FrameworkConfig(
    multi_hardware=True,
    dynamic_quant=True,
    learned_quant=True,
    quant_mode="hybrid",
    hardware_modes=("gpu", "cpu", "low_resource"),
    training_episodes=3_000,
    evaluation_episodes=400,
    continuous_training=False,
    eval_interval=1_000,
    checkpoint_interval=5_000,
    max_training_episodes=50_000,
    run_name="adaptive_universal_policy",
)
