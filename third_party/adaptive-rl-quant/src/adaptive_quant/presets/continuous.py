"""Preset for continuous RL over a streaming task sequence."""

from adaptive_quant.presets.baseline import CONFIG

CONFIG_CONTINUOUS = CONFIG.clone(
    run_name="adaptive_continuous_policy",
    continuous_learning_enabled=True,
    continuous_max_tasks=2_048,
    continuous_task_stream_mode="library_cycle",
    continuous_update_every_n_tasks=4,
    continuous_eval_every_n_tasks=256,
    continuous_checkpoint_every_n_tasks=512,
    continuous_replay_capacity=1024,
    continuous_batch_size=32,
    continuous_min_replay_before_update=8,
    continuous_drift_window=48,
    continuous_drift_reward_delta=4.0,
    continuous_exploration_rate=0.20,
    evaluation_episodes=48,
    training_episodes=1,
)
