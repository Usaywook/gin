def set_parser(parser):
    subparser = parser.add_parser('sac')

    subparser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    subparser.add_argument('--cnn_hidden', type=int, default=512, metavar='N',
                        help='cnn_hidden size (default: 512)')

    subparser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    subparser.add_argument('--temperature', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    subparser.add_argument('--entropy_sensitivity', type=float, default=0.98, metavar='G',
                           help='sensitivity of entropy for discrete action (default: 0.98)') # for cartpole, entropy sensitivity = 0.8
    subparser.add_argument('--policy', default="Stochastic",
                        help='Policy Type: Stochastic | Deterministic (default: Stochastic)')
    subparser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    subparser.add_argument('--automatic_entropy_tuning', action='store_true', default=True,
                        help='Automaically adjust α (default: False)')
    return subparser