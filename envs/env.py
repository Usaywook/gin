import gym

from envs.wrapper import DummyVecEnv, VecNormalize
from custom_tasks.envs.utils.env_wrapper import CarlaCtrMask, ActionSkipEnv
# from envs.point_gather import PointGatherEnv

def make_env(args, rank=0):
    name = args.env
    if name.startswith('Carla'):
        env = gym.make(name, args=args, carla_port=args.carla_port + rank * 1000, tm_port=args.tm_port + rank * 1000)
        env._max_episode_steps = args.max_episode_steps
        env = CarlaCtrMask(env)
        env = ActionSkipEnv(env, skip=args.skip)
    # elif name == 'point_gather':
    #     env = PointGatherEnv()
    else:
        env = gym.make(name)
        env._max_episode_steps = args.max_episode_steps

    return env


def make_venv(args, rank):
    def _func():
        env = make_env(args, rank)
        if args.seed > 0:
            env.seed(args.seed + rank)
        return env
    return _func


def make_vec_envs(args):
    envs = [make_venv(args, i) for i in range(args.num_proc)]
    envs = DummyVecEnv(envs)
    envs = VecNormalize(envs, ob=args.norm_obs, ret=args.norm_rew, scale=args.reward_scale, gamma=args.discount)
    return envs
