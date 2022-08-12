import torch
import numpy as np
from gym.spaces import Box
from torch.utils.data import DataLoader

from arguments import get_args
from envs.env import make_vec_envs
from utils import prGreen, prCyan, prYellow, prLightPurple, prRed, seed_torch, get_output_folder
from storages import TPFeeder
from algos import get_agent

np.set_printoptions(suppress=True, precision=3)
torch.set_printoptions(sci_mode=False, precision=3)

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        prCyan(args)

    if args.seed > 0:
        seed_torch(args.seed, args.cuda)
    args.output = get_output_folder(args.output, args.project, args.algo, args.env, args.seed, args.study)

    if args.algo =='grip':
        obs_space = Box(low=0, high=1, shape=(10,))
        act_space = Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), dtype=np.float32)
    else:
        envs = make_vec_envs(args)
        obs_space = envs.observation_space
        act_space = envs.action_space

    agent = get_agent(obs_space, act_space, args)

    if args.debug:
        prGreen(f'{"=" * 80} \n'
                f'\t {"Mode":<15} : {"{}".format(args.mode):<15}\n'
                f'\t {"Directory":<15} : {"{}".format(args.output):<15} \n'
                f'\t {"Device":<15} : {"{}".format(args.device):<15} \n'
                f'\t {"Algorithm":<15} : {"{}".format(args.algo).upper():<15} \n')

    if args.mode == 'test':
        try:
            data_path = 'demonstrations/CarlaNavigation/test.pt'
            weighted = args.weighted_graph if 'weighted_graph' in args else False
            test_loader = DataLoader(TPFeeder(data_path=data_path, train=False, test=True, weighted=weighted),
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False)

            agent.load_model(args.output)
            agent.eval()
            test_result = agent.validate_epoch(test_loader)
            if args.debug:
                prYellow(f'{"=" * 80}')
                for name, value in test_result.items():
                    prYellow(f'\t {name:<15} : {value:<15.2f}')
                prYellow(f'{"=" * 80}')
                np.savetxt('{}/prediction_test_log.txt'.format(args.output),
                        [list(test_result.values())],
                        delimiter=',', header=','.join(list(test_result.keys())), fmt='%.2f', comments='')

        except KeyboardInterrupt:
            prRed('KeyboardInterrupt')

        if not args.algo == 'grip':
            envs.close()