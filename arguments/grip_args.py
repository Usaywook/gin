def set_parser(parser):
    subparser = parser.add_parser('grip')
    subparser.add_argument('--trainable_edge', action='store_true', default=False)
    subparser.add_argument('--weighted_graph', action='store_true', default=False)
    subparser.add_argument('--temporal_kernel_size', type=int, default=5)
    subparser.add_argument('--graph_hidden_size', type=int, default=64)
    subparser.add_argument('--rnn_hidden_size', type=int, default=64)
    subparser.add_argument('--rnn_num_layer', type=int, default=2)
    subparser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    subparser.add_argument('--dropout', type=float, default=0.5)
    subparser.add_argument('--max_x', type=float, default=1.)
    subparser.add_argument('--max_y', type=float, default=1.)
    subparser.add_argument('--tp_lr', type=float, default=3e-4)
    subparser.add_argument('--error_order', type=int, default=1)

    subparser.add_argument('--max_hop', type=int, default=2)
    subparser.add_argument('--max_object', type=int, default=10)
    subparser.add_argument('--num_frame', type=int, default=24)
    subparser.add_argument('--neighbor_distance', type=float, default=15)
    subparser.add_argument('--neighbor_boundary', type=float, default=40)
    subparser.add_argument('--sigma', type=float, default=5)
    subparser.add_argument('--center', type=int, default=-1)

    return subparser
