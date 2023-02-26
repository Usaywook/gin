import os
from itertools import count
from collections import namedtuple

import torch
import numpy as np

import carla
import custom_tasks

from arguments import get_args
from envs.env import make_vec_envs
from utils import prGreen, prCyan, prYellow, prLightPurple, prRed, seed_torch, get_output_folder, update_linear_schedule
from utils.safety_checker import trajectory_hazard
from algos import get_agent

np.set_printoptions(suppress=True, precision=3)
torch.set_printoptions(sci_mode=False, precision=3)
Data = namedtuple('Data', ['graph_feature', 'adjacency', 'weighted_adjacency'])

def get_data(envs):
    for env in envs.venv.envs:
        env.tp_graph.weighted_graph = False
        features, adj = env.tp_graph.node_n_adacency(env.tp_neighbors)
        env.tp_graph.weighted_graph = True
        _, weighted_adj = env.tp_graph.node_n_adacency(env.tp_neighbors)
    return features, adj, weighted_adj

def test(envs, agent, args):
    total_container = {}
    for key in Data._fields:
        total_container.update({key : []})

    success_cnt = 0
    n_trial = 0
    agent.reset()
    state = envs.reset()
    episode_container = []
    for step in count():
        with torch.no_grad():
            action = agent.select_action(state)

            if args.render:
                envs.render()

            if args.use_graph and args.hazard:
                trajectory_hazard(envs, agent.predicted, agent.predicted_mask, args)

        next_state, reward, done, info = envs.step(action)

        episode_container.append(Data(*get_data(envs)))

        if 'success:episode' in info[0].keys():
            success_cnt += 1
        if np.any(done):
            n_trial += 1
            for key in Data._fields:
                total_container[key].append(np.stack(list(map(lambda x: getattr(x, key), episode_container))))
            for name, value in info[0].items():
                prYellow(f'\t {name:<15} : {value:<15.1f}')
            prCyan(f'\t number of success : {success_cnt} / {args.validate_episodes}')
            prYellow(f'{"=" * 80} \n')

        if n_trial == args.validate_episodes:
            for key in Data._fields:
                total_container[key] = torch.from_numpy(np.concatenate(total_container[key])).type(torch.float32)
            prCyan(f'{"=" * 80} \n')
            for name, value in total_container.items():
                prCyan(f'\t {name:<15} : {value.shape}')
            prCyan(f'{"=" * 80} \n')

            # path setting and save
            file_path = 'demonstrations/' + args.env.split('-')[0]
            os.makedirs(file_path, exist_ok=True)
            file_name = 'test.pt'
            file_path = os.path.join(file_path, file_name.lower())
            torch.save(total_container, file_path)
            prCyan(f'Demonstrations are saved in {file_path}')
            break

        state = next_state

    test_logs = {'success_rate' : success_cnt / (n_trial + 1)}
    return test_logs

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        prCyan(args)

    if args.seed > 0:
        seed_torch(args.seed, args.cuda)

    args.output = get_output_folder(args.output, args.project, args.algo, args.env, args.seed, args.study)

    envs = make_vec_envs(args)

    agent = get_agent(envs.observation_space, envs.action_space, args)

    if args.debug:
        prGreen(f'{"=" * 80} \n'
                f'\t {"Mode":<15} : {"{}".format(args.mode):<15}\n'
                f'\t {"Directory":<15} : {"{}".format(args.output):<15} \n'
                f'\t {"Device":<15} : {"{}".format(args.device):<15} \n'
                f'\t {"Algorithm":<15} : {"{}".format(args.algo).upper():<15} \n'
                f'\t {"Observation":<15} : {"{}".format(envs.observation_space):<15} \n'
                f'\t {"Action":<15} : {"{}".format(envs.action_space):<15} \n'
                )
    try:
        if args.mode == 'demo':
            agent.load_model(args.output, envs)
            envs.eval()
            agent.eval()
            test_logs = test(envs, agent, args)

            if args.debug:
                prLightPurple(f'{"=" * 80} \n')
                for name, value in test_logs.items():
                    prLightPurple(f'\t {name:<15} : {value:<15.2f}')
                prLightPurple(f' {"=" * 80} \n')

    except KeyboardInterrupt:
        prRed('KeyboardInterrupt')
        envs.close()

    prGreen('Complete')
    envs.close()
