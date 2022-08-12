from re import sub


def set_subparser(subparsers):
    subparser = subparsers.add_parser('cpo')

    subparser.add_argument('--policy', default="Stochastic",
                        help='Policy Type: Stochastic | Deterministic (default: Stochastic)')
    subparser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')

    subparser.add_argument('--critic_lr', type=float, default=1e-3, metavar='G',
                           help='learning rate for value (default: 1e-3)')
    subparser.add_argument('--cost_lr', type=float, default=1e-3, metavar='G',
                           help='learning rate for cost (default: 1e-3)')
    subparser.add_argument('--linear_lr', type=float, default=1e-3, metavar='G',
                           help='lbfgs learning rate (default: 1e-4)')

    subparser.add_argument('--n_epoch', type=int, default=1, metavar='N',
                        help='number of epoch for training cpo')
    subparser.add_argument('--val_iter', type=int, default=5, metavar='G',
                           help='number of iteration for value')
    subparser.add_argument('--cost_iter', type=int, default=5, metavar='G',
                           help='number of iteration for cost')
    subparser.add_argument('--cg_max_iter', type=int, default=10, metavar='G',
                           help='maximum number of iteration for conjugate gradient')
    subparser.add_argument('--line_search_max_iter', type=int, default=10, metavar='G',
                           help='maximum number of iteration for backtracking line search')

    subparser.add_argument('--lambd', type=float, default=0.97, metavar='G',
                           help='gae (default: 0.97)')
    subparser.add_argument('--lambd_c', type=float, default=0.92, metavar='G',
                           help='gae (default: 0.97)')
    subparser.add_argument('--l2_reg', type=float, default=1e-3, metavar='G',
                           help='l2 regularization regression (default: 1e-3)')
    subparser.add_argument('--max_kl', type=float, default=1e-2, metavar='G',
                           help='max kl value (default: 1e-2)')
    subparser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                           help='damping (default: 1e-1)')
    subparser.add_argument('--max_constraint', type=float, default=0.01, metavar='G',
                           help='maximum constraint value')
    subparser.add_argument('--line_search_coef', type=float, default=0.5, metavar='G',
                           help='back tracking line search coefficent')
    subparser.add_argument('--accept_ratio', type=float, default=0.5, metavar='G',
                           help='mininum acceptance ratio for back tracking line search')
    subparser.add_argument('--fusion', action='store_true', default=False)

    return subparser
