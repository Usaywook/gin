import os
from itertools import count

import torch
import numpy as np
from gym.spaces import Box
from torch.utils.data import DataLoader

from arguments import get_args
from envs import make_env
from utils import prGreen, prCyan, prYellow, prLightPurple, prRed, seed_torch, get_output_folder, update_linear_schedule
from utils import logging_time
from storages import TPFeeder
from algos import get_agent
from agents.CarlaNavigationAgent import NavigationAgent
from carla2gym.carla.util import vis_trajectory

np.set_printoptions(suppress=True, precision=3)
torch.set_printoptions(sci_mode=False, precision=3)


def get_loader(data_path, args):
    train_loader = DataLoader(TPFeeder(data_path=data_path, train=True),
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(TPFeeder(data_path=data_path, train=False),
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=False)
    return train_loader, val_loader


def is_save(eval_debugs, save_check):
    save_flag = False
    if eval_debugs['ADE_mean'] < save_check['min_ADE_mean']:
        save_check['min_ADE_mean'] = eval_debugs['ADE_mean']
        save_flag = True
    return save_flag, save_check


@logging_time(verbose=True)
def evaluate(envs, agent, args):
    navigator = NavigationAgent(envs)
    rewards = []
    times = []
    distances = []
    success_cnt = 0
    with torch.no_grad():
        for i in range(args.validate_episodes):
            navigator.reset()
            agent.reset()
            state = envs.reset()

            episode_reward = 0
            for t in count():
                action = navigator.step(state['feature'])
                predicted, predicted_mask = agent.select_action(state)

                num_object = np.sum(predicted_mask).astype(int)
                n_pred = np.transpose(predicted[0], (2, 1, 0))[:num_object]
                ego_trans = envs.actor_info['actors'][0].get_transform()
                pids = list(envs.neighbors.keys())
                vis_trajectory(envs.world, ego_trans, n_pred, pids)

                next_state, reward, done, info = envs.step(action)

                episode_reward += reward

                if args.render:
                    envs.render()

                state = next_state

                if args.max_episode_steps and t >= args.max_episode_steps - 1:
                    done = True

                if done:
                    rewards.append(episode_reward)
                    times.append(t+1)
                    if 'success:episode' in info.keys():
                        success_cnt += int(info['success:episode'] == True)
                    if 'episode_distance' in info.keys():
                        distances.append(info['episode_distance'])
                    break

    eval_logs = {'eval_mean_return': np.mean(rewards),
                'eval_std_return' : np.std(rewards),
                'eval_mean_step' : np.mean(times),
                'eval_std_step' : np.std(times),
                'success_rate' : success_cnt / args.validate_episodes}

    if len(distances) !=0:
        eval_logs.update({'eval_mean_distance' : np.mean(distances),
                         'eval_std_distance' : np.std(distances)})

    return eval_logs


if __name__ == "__main__":
    # TODO : Argument
    args = get_args()
    if args.debug:
        prCyan(args)

    if args.seed > 0:
        seed_torch(args.seed, args.cuda)
    args.output = get_output_folder(args.output, args.project, args.algo, args.env, args.seed, args.study)

    # TODO : Agent
    obs_space = Box(low=0, high=1, shape=(10,))
    act_space = Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), dtype=np.float32)
    agent = get_agent(obs_space, act_space, args)

    if args.debug:
        prGreen(f'{"=" * 80} \n'
                f'\t {"Mode":<15} : {"{}".format(args.mode):<15}\n'
                f'\t {"Directory":<15} : {"{}".format(args.output):<15} \n'
                f'\t {"Device":<15} : {"{}".format(args.device):<15} \n'
                f'\t {"Algorithm":<15} : {"{}".format(args.algo).upper():<15} \n')

    if args.mode == 'pretrain':
        save_check = {'min_ADE_mean': np.inf}

        train_loader, val_loader = get_loader('demonstrations/CarlaNavigation/rule.pt', args)
        for epoch in count():
            agent.train()
            train_losses, train_debugs = agent.train_epoch(train_loader)

            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                for name, optim in agent.get_optimizer().items():
                    init_lr = getattr(args,name+'_lr') if hasattr(args, name+'_lr') else args.lr
                    end_lr = getattr(args,name+'_end_lr') if hasattr(args, name+'_end_lr') else args.end_lr
                    update_linear_schedule(optim, epoch, args.num_epochs, init_lr, end_lr)

            if args.debug:
                prYellow(f'{"=" * 80} \n'
                        f'\t {"Epoch":<15} : {epoch:<15}'
                        f'\t {"Step":<15} : {agent.total_it:<15}')
                for name, optim in agent.get_optimizer().items():
                    prYellow(f'\t {name + "_lr":<15} : {agent.get_lr(optim):<15.8f}')
                for loss_name, loss_value in train_losses.items():
                    prYellow(f'\t {loss_name:<15} : {loss_value:<15.4f}')
                for name, value in train_debugs.items():
                    prYellow(f'\t {name:<15} : {value:<15.4f}')
                prYellow(f'{"=" * 80} \n')

            if epoch % args.validate_interval == 0 and args.validate:
                agent.eval()
                eval_debugs = agent.validate_epoch(val_loader)
                if args.debug:
                    prLightPurple(f'{"=" * 80} \n'
                                f'\t {"Eval Epoch":<15} : {"{}".format(epoch):<15}')
                    for name, value in eval_debugs.items():
                        prLightPurple(f'\t {name:<15} : {value:<15.4f}')
                    prLightPurple(f' {"=" * 80} \n')

                save_flag, save_check = is_save(eval_debugs, save_check)

                if args.write_summary:
                    agent.write_summary(losses=train_losses, train_logs=train_debugs, eval_logs=eval_debugs, web=args.wandb)

                if save_flag == True:
                    agent.save_model(args.output)
                    if args.debug:
                        prCyan(f'\t {epoch} th Epoch was saved!')
                        for name, value in save_check.items():
                            prCyan(f'\t {name:<15} : {value:<15.4f}')
                        path = '{}/save_log.txt'.format(args.output)
                        header = '' if os.path.isfile(path) else ','.join(['epoch', 'iteration', *eval_debugs.keys()])
                        with open(path, "ab") as f:
                            np.savetxt(f, [[epoch, agent.total_it, *eval_debugs.values()]],
                                            delimiter=',',
                                            header=header,
                                            fmt='%.2f', comments='')

            if epoch >= args.num_epochs - 1:
                break

    elif args.mode == 'test':
        # TODO : Vectorized Environment
        envs = make_env(args)
        if args.debug:
            prGreen(f'\t {"Observation":<15} : {"{}".format(obs_space):<15} \n'
                    f'\t {"Action":<15} : {"{}".format(act_space):<15} \n')
        try:
            agent.eval()
            agent.load_model(args.output, envs)

            eval_logs = evaluate(envs, agent, args, verbose=True, slience=False)

            if args.debug:
                for name, value in eval_logs.items():
                    prYellow(f'\t {name:<15} : {value:<15.2f}')
                np.savetxt('{}/test_log.txt'.format(args.output),
                        [list(eval_logs.values())],
                        delimiter=',', header=','.join(list(eval_logs.keys())), fmt='%.2f', comments='')

            envs.close()

        except KeyboardInterrupt:
            prRed('KeyboardInterrupt')
            envs.close()

    else:
        prRed('undefined mode {}'.format(args.mode))


    prGreen('Complete')