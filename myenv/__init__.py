from gym.envs.registration import register

register(
    id='RotaryPend-v0',
    entry_point='myenv.envs:RotaryPendulumEnv',
    max_episode_steps=800,
)

register(
    id='RotaryPend-v1',
    entry_point='myenv.envs:RotaryPendulumEnv',
    max_episode_steps=2000,
)

