from gym.envs.registration import register

register(
    id='CarlaNavigation-v0',
    entry_point='custom_tasks.envs:CarlaNavigationEnv',
    max_episode_steps=1000,
    reward_threshold=-110.0
)
register(
    id='CarlaDetour-v0',
    entry_point='custom_tasks.envs:CarlaDetourEnv',
    max_episode_steps=1000,
    reward_threshold=-110.0
)
register(
    id='CarlaFollowing-v0',
    entry_point='custom_tasks.envs:CarlaFollowingEnv',
    max_episode_steps=200,
    reward_threshold=-110.0
)
