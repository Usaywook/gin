import argparse
import os
import gym
import custom_tasks
import torch
from agents.CarlaDetourAgent import DetourAgent
import random
import numpy as np
from utils.env_wrapper import CarlaCtrMask

def get_demo(args):
    print(args)
    env = gym.make(args.game)
    env = CarlaCtrMask(env)

    obs_space = env.observation_space
    act_space = env.action_space.shape
    print('observation space: {}'.format(obs_space))
    print('action space: {}'.format(act_space))
    if args.use_dict_obs:
        state_space = obs_space['feature'].shape
        if args.use_camera:
            camera_space = obs_space['camera'].shape
        if args.use_birdeye:
            birdeye_space = obs_space['birdeye'].shape
        if args.use_lidar:
            lidar_space = obs_space['lidar'].shape
    else:
        state_space = obs_space.shape

    agent = DetourAgent(env)

    n_trial = 1

    if args.save:
        # trajectory container
        traj_states = []
        traj_actions = []
        traj_rewards = []
        traj_lengths = []
        if args.use_dict_obs:
            if args.use_camera:
                traj_camera = []
            if args.use_birdeye:
                traj_birdeye = []
            if args.use_lidar:
                traj_lidar = []
        success_cnt = 0
        n_trial = 5

    if args.check and not args.save:
        check = []
    try:
        for epi in range(args.n_episode * n_trial):
            print("reset environment...")
            obs = env.reset()
            agent.reset() # for version 1
            print("=" * 30 + "{}th episode".format(epi + 1) + "=" * 30)

            if args.save:
                save_flag = False
                # reset episode container
                states_container = np.zeros((args.max_episode_step, state_space[-1]), dtype=np.float32)
                if args.use_dict_obs:
                    if args.use_camera:
                        camera_container = torch.zeros((args.max_episode_step, *camera_space), dtype=torch.uint8)
                    if args.use_birdeye:
                        birdeye_container = torch.zeros((args.max_episode_step, *birdeye_space), dtype=torch.uint8)
                    if args.use_lidar:
                        lidar_container = torch.zeros((args.max_episode_step, *lidar_space), dtype=torch.uint8)
                actions_container = np.zeros((args.max_episode_step, act_space[-1]), dtype=np.float32)
                rewards_container = np.zeros((args.max_episode_step, 1), dtype=np.float32)

            if args.debug:
                rewards_container = np.zeros((args.max_episode_step, 1), dtype=np.float32)

            for step in range(args.max_episode_step):

                if args.use_dict_obs:
                    action = agent.step(obs['feature'])
                else:
                    action = agent.step(obs)

                next_obs, reward, done, info = env.step(action)

                if args.check and not args.save:
                    if args.use_dict_obs:
                        check.append(next_obs['feature'])
                    else:
                        check.append(next_obs)

                if args.save:
                    if args.use_dict_obs:
                        states_container[step] = obs['feature']
                        if args.use_camera:
                            camera_container[step] = torch.from_numpy(obs['camera']).type(torch.uint8)
                        if args.use_birdeye:
                            birdeye_container[step] = torch.from_numpy(obs['birdeye']).type(torch.uint8)
                        if args.use_lidar:
                            lidar_container[step] = torch.from_numpy(obs['lidar']).type(torch.uint8)
                    else:
                        states_container[step] = obs
                    actions_container[step] = action
                    rewards_container[step] = reward

                    # TODO: filter for demo quality
                    if done and step <= (args.max_episode_step - 1) and 'success:episode' in info:
                        save_flag = True
                        print("episode reward is {:5f}, {} episode is saved!".format(rewards_container.sum(), success_cnt+1))

                if args.debug:
                    rewards_container[step] = reward

                obs = next_obs
                if done or step == (args.max_episode_step - 1):
                    print("episode is finished after {} time step".format(step + 1))

                    if step < args.max_episode_step:
                        print("reason : {} ".format(info))

                    if args.save:
                        # TODO: if expert quality is satisfied, save it
                        if save_flag:
                            success_cnt += 1
                            traj_states.append(states_container)
                            traj_actions.append(actions_container)
                            traj_rewards.append(rewards_container)
                            traj_lengths.append(step + 1)
                            if args.use_dict_obs:
                                if args.use_camera:
                                    traj_camera.append(camera_container)
                                if args.use_birdeye:
                                    traj_birdeye.append(birdeye_container)
                                if args.use_lidar:
                                    traj_lidar.append(lidar_container)

                    if args.debug:
                        print("episode finisihed reason: {}, episode reward is {:5f}".format(info, rewards_container.sum()))

                    if args.check:
                        if args.save:
                            # save state_seq as csv format
                            # np.savetxt('/home/usaywook/Documents/state_seq.csv', np.array(traj_states), delimiter=',')
                            print("state max: {}".format(np.array(traj_states).max(axis=0)))
                            print("state min: {}".format(np.array(traj_states).min(axis=0)))
                        else:
                            # save state_seq as csv format
                            # np.savetxt('/home/usaywook/Documents/state_seq.csv', np.array(check), delimiter=',')
                            print("state max: {}".format(np.array(check).max(axis=0)))
                            print("state min: {}".format(np.array(check).min(axis=0)))
                            check = []
                    break

            if args.save:
                if success_cnt >= args.n_episode:
                    print("Success Rate is {}".format(success_cnt / (epi + 1) * 100))
                    break
    except KeyboardInterrupt:
        print("Force quit!")
        env.close()
    env.close()

    if args.save:
        expert_demo = {'states': torch.Tensor(traj_states),
                       'actions': torch.Tensor(traj_actions),
                       'rewards': torch.Tensor(traj_rewards),
                       'lengths': torch.Tensor(traj_lengths).type(torch.LongTensor)}
        if args.use_dict_obs:
            if args.use_camera:
                expert_demo.update({'camera': torch.stack(traj_camera, dim=0)})
            if args.use_birdeye:
                expert_demo.update({'birdeye': torch.stack(traj_birdeye, dim=0)})
            if args.use_lidar:
                expert_demo.update({'lidar': torch.stack(traj_lidar, dim=0)})

        if args.use_dict_obs:
            if args.use_camera:
                expert_demo['camera'] = expert_demo['camera'].permute(0, 1, 4, 2, 3).type(torch.FloatTensor)/255.
                print('camera states: {}'.format(expert_demo['camera'].shape))
            if args.use_birdeye:
                expert_demo['birdeye'] = expert_demo['birdeye'].permute(0, 1, 4, 2, 3).type(torch.FloatTensor)/255.
                print('birdeye states: {}'.format(expert_demo['birdeye'].shape))
            if args.use_lidar:
                expert_demo['lidar'] = expert_demo['lidar'].permute(0, 1, 4, 2, 3).type(torch.FloatTensor)/255.
                print('lidar states: {}'.format(expert_demo['lidar'].shape))

        print('states: {}'.format(expert_demo['states'].shape))
        print('actions: {}'.format(expert_demo['actions'].shape))
        print('rewards: {}'.format(expert_demo['rewards'].shape))
        print('lengths: {}'.format(expert_demo['lengths'].shape))

        # path setting and save
        file_path = args.save_path + '/' + args.game.split('-')[0]
        os.makedirs(file_path, exist_ok=True)
        file_name = "trajs_" + args.game.split('-')[0] + "_" + args.algo + '.pt'
        file_path = os.path.join(file_path, file_name.lower())
        torch.save(expert_demo, file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='CarlaDetour-v0',
                        help='CarlaCompound-v0|CarlaDetour-v0')
    parser.add_argument('--algo', type=str, default='rb2')
    parser.add_argument('-ms', '--max-episode-step', type=int, default=1000)
    parser.add_argument('-ne', '--n-episode', type=int, default=2)
    parser.add_argument('--save_path', type=str, default='../gail_experts', help='Where to save expert trajectory')
    parser.add_argument('--save', action='store_true', default=False, help='save expert demonstration')
    parser.add_argument('--check', action='store_true', default=False, help='check state value range')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--debug', action='store_true', default=True, help='check reward value')
    parser.add_argument('--use-dict-obs', action='store_true', default=False, help='use dictionary for obs space')
    parser.add_argument('--use-camera', action='store_true', default=False, help='use raw camera for obs space')
    parser.add_argument('--use-birdeye', action='store_true', default=False, help='use birdeye for obs space')
    parser.add_argument('--use-lidar', action='store_true', default=False, help='use lidar for obs space')
    parser.add_argument('--carla-port', type=int, default=2000)
    parser.add_argument('--tm-port', type=int, default=8000)
    args = parser.parse_args()
    get_demo(args)