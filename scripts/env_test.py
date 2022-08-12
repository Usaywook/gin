import gym
from gym import envs
from collections import namedtuple
from custom_tasks.envs.utils.env_wrapper import CarlaCtrMask, CarlaLocalPlannerMask
from agents.CarlaNavigationAgent import NavigationAgent
import custom_tasks

# print(gym.__file__)
# envids = [spec.id for spec in envs.registry.all()]
# for envid in sorted(envids):
#     print(envid)
args_dict = {'carla_ip': 'localhost',
             'use_graph': False,
             'use_dict_obs': True,
             'traffic_rate': 0.,
             'seed': -1}
Arguments = namedtuple('Arguments', args_dict.keys())
args = Arguments(**args_dict)

# env = gym.make('CartPole-v0')
# env = gym.make('CarlaNavigation-v0')
# env = gym.make('CarlaDetour-v0')
env = gym.make('CarlaFollowing-v0', args=args, carla_port=4000, tm_port=4050)
env = CarlaCtrMask(env)
# env = CarlaLocalPlannerMask(env)

# env = gym.make('Pendulum-v0')
print(env.observation_space)
print(env.action_space)

for i in range(3):
    obs = env.reset()
    print("reset environment...")
    agent = NavigationAgent(env)
    agent.reset()
    print("="*30 + "{}th episode".format(i) + "="*30)
    for t in range(100):
        # env.render()
        # action = env.action_space.sample()
        if args.use_dict_obs:
            action = agent.step(obs['feature'])
        else:
            action = agent.step(obs)

        obs, reward, done, info = env.step(action)
        # print("obs =",obs)
        # print("reward =",reward)
        # print("done =", done)
        # print("info =", info)
        if done or t == 99:
            print("episode is finished after {} time step".format(t+1))
            if t < 100:
                print("reason : {} ".format(info))
            break
env.close()
