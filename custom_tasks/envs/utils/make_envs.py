from utils.env_wrapper import *

def make_env(env_name):
    env = gym.make(env_name)
    if len(env.observation_space.shape) > 1:
        env = MaxAndSkipEnv(env)
        env = FireResetEnv(env)
        env = ProcessFrame84(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 4)
        env = ScaledFloatFrame(env)
    else:
        env = NormalizedEnv(env)
    return env