def set_subparser(subparsers):
    subparser = subparsers.add_parser('replay')
    subparser.add_argument('--buffer_type', type=str, default='vanilla', help='vanilla | per | n_step | n_stepper')

    # vanilla
    subparser.add_argument('--rmsize', type=int, default=50000, metavar='N',
                        help='size of replay buffer (default: 50000)')

    # per
    subparser.add_argument('--alpha', default=0.6, type=float)
    subparser.add_argument('--beta', default=0.4, type=float)
    subparser.add_argument('--e', default=1e-6)
    subparser.add_argument('--beta_increment_per_sampling', default=1e-5, type=float)

    # nstep
    subparser.add_argument('--n_step', default=3, type=int)

    return subparser