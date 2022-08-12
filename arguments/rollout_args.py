def set_subparser(subparsers):
    subparser = subparsers.add_parser('rollout')

    subparser.add_argument("--rollout_length", default=2000, type=int)
    subparser.add_argument('--buffer_type', type=str, default='vanilla', help='vanilla')

    return subparser