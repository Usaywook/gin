def set_subparser(subparsers):
    subparser = subparsers.add_parser('trpo')

    subparser.add_argument('--storage_type', type=str, default='rollout')
    subparser.add_argument("--rollout_length", default=2048, type=int)
    subparser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                           help='hidden size (default: 256)')


    subparser.add_argument('--lambd', type=float, default=0.97, metavar='G',
                           help='gae (default: 0.97)')
    subparser.add_argument('--l2_reg', type=float, default=1e-3, metavar='G',
                           help='l2 regularization regression (default: 1e-3)')
    subparser.add_argument('--max_kl', type=float, default=1e-2, metavar='G',
                           help='max kl value (default: 1e-2)')
    subparser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                           help='damping (default: 1e-1)')
    subparser.add_argument('--max_backtrack_step', type=int, default=10, metavar='G',
                           help='maximum back tracking line search step (default: 10)')
    subparser.add_argument('--critic_lr', type=float, default=1e-3, metavar='G',
                           help='lbfgs learning rate (default: 1e-3)')
    subparser.add_argument('--linear_lr', type=float, default=1e-3, metavar='G',
                           help='lbfgs learning rate (default: 1e-3)')

    subparser.add_argument('--policy', default="Stochastic",
                        help='Policy Type: Stochastic | Deterministic (default: Stochastic)')
    subparser.add_argument('--n_epoch', type=int, default=1)

    return subparser