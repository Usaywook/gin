import os
from itertools import count
from collections import namedtuple

import torch
import numpy as np

from arguments import get_args
from envs import make_env
from utils import prGreen, prCyan, prYellow, prLightPurple, prRed, seed_torch
from utils import logging_time  #,logging_sampling_peak_memory
from agents.CarlaNavigationAgent import NavigationAgent

np.set_printoptions(suppress=True, precision=3)
torch.set_printoptions(sci_mode=False, precision=3)
Data = namedtuple('Data', ['graph_feature', 'adjacency'])

# @logging_sampling_peak_memory
@logging_time(verbose=True)
def evaluate(envs, agent, args):
    rewards = []
    times = []
    distances = []
    total_container = {}
    for key in Data._fields:
        total_container.update({key : []})
    success_cnt = 0
    with torch.no_grad():
        for n_trial in count():
            agent.reset()
            state = envs.reset()
            episode_container = []
            episode_reward = 0
            for t in count():
                action = agent.step(state['feature'])

                next_state, reward, done, info = envs.step(action)

                episode_container.append(Data(*envs.tp_graph.node_n_adacency(envs.tp_neighbors)))

                episode_reward += reward

                if args.render:
                    envs.render()

                state = next_state

                if 'max_episode_steps' in args and t >= args.max_episode_steps - 1:
                    done = True

                if done:
                    rewards.append(episode_reward)
                    times.append(t+1)
                    if 'success:episode' in info.keys():
                        success_cnt += int(info['success:episode'] == True)
                        for key in Data._fields:
                            total_container[key].append(np.stack(list(map(lambda x: getattr(x, key), episode_container))))

                    if 'episode_distance' in info.keys():
                        distances.append(info['episode_distance'])

                    for name, value in info.items():
                        prYellow(f'\t {name:<15} : {value:<15.1f}')

                    prCyan(f'\t number of success : {success_cnt} / {args.validate_episodes}')
                    prYellow(f'{"=" * 80} \n')
                    break

            if success_cnt == args.validate_episodes:
                for key in Data._fields:
                    total_container[key] = torch.from_numpy(np.concatenate(total_container[key])).type(torch.float32)
                prCyan(f'{"=" * 80} \n')
                for name, value in total_container.items():
                    prCyan(f'\t {name:<15} : {value.shape}')
                prCyan(f'{"=" * 80} \n')

                # path setting and save
                file_path = 'demonstrations/' + args.env.split('-')[0]
                os.makedirs(file_path, exist_ok=True)
                file_name = 'rule.pt'
                file_path = os.path.join(file_path, file_name.lower())
                torch.save(total_container, file_path)
                prCyan(f'Demonstrations are saved in {file_path}')
                break

    eval_logs = {'eval_mean_return': np.mean(rewards),
                'eval_std_return' : np.std(rewards),
                'eval_mean_step' : np.mean(times),
                'eval_std_step' : np.std(times),
                'success_rate' : success_cnt / (n_trial + 1)}

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

    # TODO : Vectorized Environment
    envs = make_env(args)

    # TODO : Agent
    agent = NavigationAgent(envs)

    if args.debug:
        prGreen(f'{"=" * 80} \n'
                f'\t {"Mode":<15} : {"{}".format(args.mode):<15}\n'
                f'\t {"Device":<15} : {"{}".format(args.device):<15} \n'
                f'\t {"Observation":<15} : {"{}".format(envs.observation_space):<15} \n'
                f'\t {"Action":<15} : {"{}".format(envs.action_space):<15} \n'
                )

    try:
        if args.mode == 'demo':
            eval_logs = evaluate(envs, agent, args, verbose=True, slience=False)

            if args.debug:
                prLightPurple(f'{"=" * 80} \n')
                for name, value in eval_logs.items():
                    prLightPurple(f'\t {name:<15} : {value:<15.2f}')
                prLightPurple(f' {"=" * 80} \n')

    except KeyboardInterrupt:
        prRed('KeyboardInterrupt')
        envs.close()

    prGreen('Complete')
    envs.close()
