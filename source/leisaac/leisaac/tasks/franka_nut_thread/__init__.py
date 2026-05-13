import gymnasium as gym

gym.register(
    id="LeIsaac-Franka-NutThread-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_nut_thread_env_cfg:FrankaNutThreadEnvCfg",
    },
)
