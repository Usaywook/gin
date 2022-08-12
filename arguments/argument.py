import torch

import argparse
from arguments.gin_args import set_parser as gin_parser
from arguments.grip_args import set_parser as grip_parser
from arguments.sac_args import set_parser as sac_parser
from arguments.wcsac_args import set_parser as wcsac_parser
from arguments.cpo_args import set_subparser as cpo_parser
from arguments.trpo_args import set_subparser as trpo_parser
from arguments.replay_args import set_subparser as replay_parser
from arguments.rollout_args import set_subparser as rollout_parser

def get_arguments():
    parser = argparse.ArgumentParser(description='Base Parser')

    parser.add_argument('--mode', default='train', type=str, help='pretrain | train | test')
    parser.add_argument('--env', default='CarlaNavigation-v0', type=str)
    parser.add_argument('--render', dest='render', action='store_true', default=False)
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--cuda', dest='cuda', action='store_true', default=True)
    parser.add_argument('--cuda_idx', default=0, type=int, help='')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--load', dest='load', action='store_true', default=False)
    parser.add_argument('--resume', dest='resume', action='store_true', default=False)
    parser.add_argument('--resume_id', default=None, type=str)
    parser.add_argument('--resume_step', default=0, type=int)
    parser.add_argument('--resume_date', default=None, type=str)
    parser.add_argument('--validate', dest='validate', action='store_true', default=True)
    parser.add_argument('--validate_interval', default=5, type=int, help='')
    parser.add_argument('--validate_episodes', default=10, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--num_steps', default=500000, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--max_episode_steps', default=1000, type=int, help='')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true', default=True)
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)

    parser.add_argument('--algo', type=str, default='gin')
    parser.add_argument('--study', type=str, default=None)
    parser.add_argument('--write_summary', default=False, action='store_true')
    parser.add_argument('--wandb', default=True, action='store_true')
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--date', default=False, action='store_true')

    parser.add_argument('--tm_port', default=2050, type=int, help='')
    parser.add_argument('--carla_ip', default='localhost', type=str)
    parser.add_argument('--carla_port', default=2000, type=int, help='')

    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--end_lr', default=1e-4, type=float, help='end learning rate')
    parser.add_argument('--use_linear_lr_decay', default=True, action='store_true')
    parser.add_argument("--start_timesteps", default=5000, type=int, help='Time steps initial random policy is used')
    parser.add_argument('--norm_obs', default=False, action='store_true')
    parser.add_argument('--norm_rew', default=False, action='store_true')
    parser.add_argument("--batch_size", default=500, type=int, help='Batch size for both actor and critic')
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--discount", default=0.99, type=float, help='Discount factor')
    parser.add_argument("--reward_scale", default=1, type=float, help='Scale factor for reward')
    parser.add_argument('--penalty', dest='penalty', action='store_true', default=False)

    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--save_video_interval', type=int, default=100)
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--use_graph', action='store_true', default=False)
    parser.add_argument('--use_tp_graph', action='store_true', default=False)
    parser.add_argument('--use_tracking', action='store_true', default=False)
    parser.add_argument('--major', type=float, default=0.8)
    parser.add_argument('--minor', type=float, default=0.2)
    parser.add_argument('--traffic_rate', type=float, default=1.0)

    subparsers = parser.add_subparsers(dest='pg', help='grip | sac | wcsac | cpo | trpo')
    subparser = grip_parser(subparsers)
    for pg in ['sac', 'wcsac', 'cpo', 'trpo']:
        subparser = globals()['{}_parser'.format(pg)](subparsers)

        subsubparsers= subparser.add_subparsers(dest='storage', help='replay | rollout')
        for storage in ['replay', 'rollout']:
            subsubparser = globals()['{}_parser'.format(storage)](subsubparsers)

            subsubsubparsers = subsubparser.add_subparsers(dest='gse', help='gin')
            subsubsubparser = gin_parser(subsubsubparsers)

    args = parser.parse_args()

    args.device = torch.device("cuda:{}".format(args.cuda_idx) if torch.cuda.is_available() else "cpu") if args.cuda else torch.device("cpu")
    if 'storage' in args and args.storage == 'rollout':
        args.rollout_length = int(args.rollout_length / args.skip)

    return args
