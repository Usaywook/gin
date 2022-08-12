import os
from itertools import count

import torch
import numpy as np

import custom_tasks
from arguments import get_args
from envs.env import make_vec_envs
from storages import get_storage
from utils import prGreen, prCyan, prYellow, prLightPurple, prRed, seed_torch, get_output_folder, update_linear_schedule
from utils import logging_time  #,logging_sampling_peak_memory
from utils.safety_checker import trajectory_hazard
from algos import get_agent

np.set_printoptions(suppress=True, precision=3)
torch.set_printoptions(sci_mode=False, precision=3)

# @logging_sampling_peak_memory
@logging_time(verbose=True)
def train_episode(envs, agent, args, storage):
    envs.train()
    agent.train()
    agent.reset()
    episode_reward = 0
    episode_cost = 0
    mean_losses = {}

    state = envs.reset()
    for t in count():
        # agent pick action ...
        # Select action randomly or according to policy
        if agent.total_it < args.start_timesteps:
            action = envs.action_space.sample()
        else:
            action = agent.select_action(state)

            if args.use_graph and agent.total_it > args.tp_warmup:
                if args.hazard:
                    hazards = trajectory_hazard(envs, agent.predicted, agent.predicted_mask, args)

        if args.render:
            envs.render()

        # env response with next_observation, reward, terminate_info
        next_state, reward, done, info = envs.step(action)
        cost = np.array([inf['constraint_cost'] if 'constraint_cost' in inf.keys() else 0 for inf in info])

        # agent observe and update policy
        storage['buffer'].add(state, action, reward, next_state, done, cost)
        if args.use_tp_graph:
            storage['tp_buffer'].add(envs.venv.get_tp_graph())

        # update
        losses, debugs = agent.update(storage)

        for k,v in losses.items():
            mean_losses[k] = mean_losses[k] + v if k in mean_losses else v

        episode_reward += reward.item()
        episode_cost += cost
        state = next_state

        if done:  # end of episode
            break

    mean_losses = {k: v / (t + 1) for k,v in mean_losses.items()}

    episode_distance = np.array([inf['episode_distance'] if 'episode_distance' in inf.keys() else 0 for inf in info])

    train_logs = {'train_return': episode_reward,
                  'train_cost': episode_cost.mean(),
                  'train_step': t + 1,
                  'train_distance': episode_distance.mean()}

    return train_logs, mean_losses, debugs

@logging_time(verbose=True)
def train_rollout(envs, agent, args, storage):
    agent.train()
    agent.reset()
    envs.train()
    state = envs.reset()
    storage['buffer'].reset()

    distances = []
    success_cnt = 0
    for step in range(args.rollout_length):
        # agent pick action ...
        action, log_prob = agent.select_action(state)

        if args.use_graph and agent.total_it > args.tp_warmup:
            if args.hazard:
                hazards = trajectory_hazard(envs, agent.predicted, agent.predicted_mask, args)

        if args.render:
            envs.render()

        # env response with next_observation, reward, terminate_info`
        next_state, reward, done, info = envs.step(action)
        cost = np.array([inf['constraint_cost'] if 'constraint_cost' in inf.keys() else 0 for inf in info])
        distances.extend([inf['episode_distance'] for inf in np.array(info)[done] if 'episode_distance' in inf.keys()])
        success_cnt += np.sum([int(inf['success:episode'] == True) for inf in np.array(info)[done] if 'success:episode' in inf.keys()])

        if step == args.rollout_length - 1:
            done = np.ones(args.num_proc, dtype=bool)

        if args.use_tp_graph:
            state.update(envs.venv.get_tp_graph())

        storage['buffer'].add(state, action, reward, done, cost, log_prob)

        state = next_state

    train_logs = storage['buffer'].get_summary()

    train_logs.update({'train_success': success_cnt / train_logs['num_episode']})
    if len(distances) != 0:
        train_logs.update({'train_distance': np.mean(distances)})

    # update
    losses, debugs = agent.update(storage)

    return train_logs, losses, debugs

# @logging_sampling_peak_memory
@logging_time(verbose=True)
def evaluate(envs, agent, args):
    agent.reset()
    state = envs.reset()

    rewards, costs, dones, steps, distances, success_steps = [], [], [], [], [], []
    success_cnt, episode, collision_cnt = 0, 0, 0
    step = np.zeros(args.num_proc, dtype=int)
    while episode < args.validate_episodes:
        with torch.no_grad():
            action = agent.select_action(state)
            if args.render:
                envs.render()

            if args.use_graph:
                if args.hazard and (agent.total_it > args.tp_warmup or args.mode == 'test'):
                    hazards = trajectory_hazard(envs, agent.predicted, agent.predicted_mask, args)

        next_state, reward, done, info = envs.step(action)
        cost = np.array([inf['constraint_cost'] if 'constraint_cost' in inf.keys() else 0 for inf in info])

        step += np.ones(args.num_proc, dtype=int)
        if np.any(done):
            terminate_indices = np.where(done)[0]
            steps.extend(step[terminate_indices])
            success_masks = np.array([inf['success:episode'] == True if 'success:episode' in inf.keys() else False for inf in np.array(info)[done]])
            success_steps.extend(step[np.where(success_masks)[0]])
            step[terminate_indices] = 0

        rewards.append(reward)
        costs.append(cost)
        dones.append(done)
        episode += int(done.sum())
        distances.extend([inf['episode_distance'] for inf in np.array(info)[done] if 'episode_distance' in inf.keys()])
        success_cnt += np.sum([int(inf['success:episode'] == True) for inf in np.array(info)[done] if 'success:episode' in inf.keys()])
        collision_cnt += np.sum([int(inf['failure:collision'] == True) for inf in np.array(info)[done] if 'failure:collision' in inf.keys()])

        state = next_state

    dones = np.array(dones).T
    rewards = np.array(rewards).T
    rewards = [sum(arr) for i in range(args.num_proc) for arr in np.split(rewards[i], np.where(dones[i])[0] + 1)[:-1]]
    costs = np.array(costs).T
    costs = [sum(arr) for i in range(args.num_proc) for arr in np.split(costs[i], np.where(dones[i])[0] + 1)[:-1]]

    eval_logs = {'eval_mean_return': np.mean(rewards),
                'eval_std_return' : np.std(rewards),
                'eval_mean_step' : np.mean(steps),
                'eval_std_step' : np.std(steps),
                'success_rate' : success_cnt / args.validate_episodes,
                'collision_rate' : collision_cnt / args.validate_episodes}

    if len(success_steps) != 0:
        eval_logs.update({'eval_mean_success_step' : np.mean(success_steps),
                          'eval_std_success_step' : np.std(success_steps)})

    if len(distances) !=0:
        eval_logs.update({'eval_mean_distance' : np.mean(distances),
                          'eval_std_distance' : np.std(distances)})
    if len(costs) !=0:
        eval_logs.update({'eval_mean_cost' : np.mean(costs),
                          'eval_std_cost' : np.std(costs)})
    return eval_logs


def is_save(eval_logs, save_check):
    save_flag = False
    if 'eval_mean_distance' in eval_logs.keys():
        if eval_logs['eval_mean_distance'] >= save_check['max_eval_mean_distance'] or \
            (eval_logs['success_rate'] >= save_check['max_success_rate'] and eval_logs['success_rate'] != 0):
            save_flag = True
            save_check['max_eval_mean_return'] = eval_logs['eval_mean_return']
            save_check['min_eval_std_return'] = eval_logs['eval_std_return']
            save_check['min_eval_mean_cost'] = eval_logs['eval_mean_cost']
            save_check['min_eval_std_cost'] = eval_logs['eval_std_cost']
            save_check['max_eval_mean_distance'] = eval_logs['eval_mean_distance']
            save_check['min_eval_std_distance'] = eval_logs['eval_std_distance']
            save_check['max_success_rate'] = eval_logs['success_rate']
    else:
        if eval_logs['eval_mean_return'] > save_check['max_eval_mean_return']:
            save_flag = True
            save_check['max_eval_mean_return'] = eval_logs['eval_mean_return']
            save_check['min_eval_std_return'] = eval_logs['eval_std_return']

    return save_flag, save_check


if __name__ == "__main__":
    # TODO : Argument
    args = get_args()
    if args.debug:
        prCyan(args)

    if args.seed > 0:
        seed_torch(args.seed, args.cuda)

    args.output = get_output_folder(args.output, args.project, args.algo, args.env, args.seed, args.study)

    # TODO : Vectorized Environment
    envs = make_vec_envs(args)

    # TODO : Agent
    agent = get_agent(envs.observation_space, envs.action_space, args)

    if args.debug:
        prGreen(f'{"=" * 80} \n'
                f'\t {"Mode":<15} : {"{}".format(args.mode):<15}\n'
                f'\t {"Directory":<15} : {"{}".format(args.output):<15} \n'
                f'\t {"Device":<15} : {"{}".format(args.device):<15} \n'
                f'\t {"Policy":<15} : {"{}/{}".format(args.pg, args.policy).upper():<15} \n'
                f'\t {"Algorithm":<15} : {"{}".format(args.algo).upper():<15} \n'
                f'\t {"Observation":<15} : {"{}".format(envs.observation_space):<15} \n'
                f'\t {"Action":<15} : {"{}".format(envs.action_space):<15} \n'
                )

    try:
        if args.mode == 'train':
            save_check = {'max_eval_mean_return': -np.inf,
                        'min_eval_std_return': np.inf,
                        'max_success_rate': 0,
                        'max_eval_mean_distance': 0,
                        'min_eval_std_distance': np.inf,
                        'min_eval_mean_cost': np.inf,
                        'min_eval_std_cost': np.inf}

            # TODO: load model : read save_log.txt update iteration and save_check values
            if args.load or args.resume:
                agent.load_model(args.output, envs, save_check, args.resume_id, args.resume_step)
            else:
                file = '{}/save_log.txt'.format(args.output)
                if os.path.isfile(file):
                    os.remove(file)

            # TODO: Buffer
            storage = get_storage(envs.observation_space, envs.action_space, args)

            if args.algo == 'gin' and args.study == 'hmm' and not args.resume:
                prYellow(f'{"=" * 80} \n'
                f'\t Start Pretrain HMM-Encoder')
                agent.fit_encoder(args.tp_data_path)
                prYellow(f'{"=" * 80} \n')

            for i in count():
                if args.storage == 'rollout':
                    train_logs, losses, debugs = train_rollout(envs, agent, args, storage, verbose=True, slience=False)
                else:
                    train_logs, losses, debugs = train_episode(envs, agent, args, storage, verbose=True, slience=False)

                if args.use_linear_lr_decay:
                    # decrease learning rate linearly
                    for name, optim in agent.get_optimizer().items():
                        init_lr = getattr(args,name+'_lr') if hasattr(args, name+'_lr') else args.lr
                        end_lr = getattr(args,name+'_end_lr') if hasattr(args, name+'_end_lr') else args.end_lr
                        update_linear_schedule(optim, agent.total_it, args.num_steps, init_lr, end_lr)

                if args.debug:
                    prYellow(f'{"=" * 80} \n'
                            f'\t {"Train Epoch":<15} : {"{}".format(i):<15}'
                            f'\t {"Train Step":<15} : {"{} / {}".format(agent.total_it, args.num_steps):<15}')
                    for name, value in train_logs.items():
                        prYellow(f'\t {name:<15} : {value:<15.2f}')
                    prYellow(f'{"-" * 80}')
                    for name, optim in agent.get_optimizer().items():
                        prYellow(f'\t {name + "_lr":<15} : {agent.get_lr(optim):<15.8f}')
                    for loss_name, loss_value in losses.items():
                        prYellow(f'\t {loss_name:<15} : {loss_value:<15.4f}')
                    prYellow(f'{"-" * 80}')
                    for name, value in debugs.items():
                        prYellow(f'\t {name:<15} : {value:<15.4f}')
                    prYellow(f'{"=" * 80} \n')

                if i % args.validate_interval == 0 and args.validate:
                    envs.eval()
                    agent.eval()
                    args.record = i % args.save_video_interval == 0 and args.save_video

                    eval_logs = evaluate(envs, agent, args, verbose=True, slience=False)

                    if args.debug:
                        prLightPurple(f'{"=" * 80} \n'
                                    f'\t {"Eval Epoch":<15} : {"{}".format(i):<15}')
                        for name, value in eval_logs.items():
                            prLightPurple(f'\t {name:<15} : {value:<15.2f}')
                        prLightPurple(f' {"=" * 80} \n')

                    save_flag, save_check = is_save(eval_logs, save_check)

                    if args.write_summary:
                        agent.write_summary(losses=losses, train_logs=train_logs, eval_logs=eval_logs, web=args.wandb)

                    if save_flag == True:
                        agent.save_model(args.output, envs)
                        if args.debug:
                            prCyan('\t {} th Epoch was saved!, Max episode return : {} / {}'.format(i, save_check['max_eval_mean_return'], save_check['min_eval_std_return']))
                            path = '{}/save_log.txt'.format(args.output)
                            header = '' if os.path.isfile(path) else ','.join(['iteration',*save_check.keys()])
                            with open(path, "ab") as f:
                                np.savetxt(f, [[agent.total_it, *save_check.values()]],
                                                delimiter=',',
                                                header=header,
                                                fmt='%.2f', comments='')

                if agent.total_it > args.num_steps:
                    break

        elif args.mode == 'test':
            envs.eval()
            agent.eval()

            agent.load_model(args.output, envs)

            eval_logs = evaluate(envs, agent, args, verbose=True, slience=False)

            if args.debug:
                for name, value in eval_logs.items():
                    prYellow(f'\t {name:<15} : {value:<15.2f}')
                np.savetxt('{}/test_log.txt'.format(args.output),
                        [list(eval_logs.values())],
                        delimiter=',', header=','.join(list(eval_logs.keys())), fmt='%.2f', comments='')
        else:
            prRed('undefined mode {}'.format(args.mode))

    except KeyboardInterrupt:
        prRed('KeyboardInterrupt')
        envs.close()

    prGreen('Complete')
    envs.close()