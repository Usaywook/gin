from torch.utils.data import DataLoader

from utils.misc import prRed
from storages.replaybuffer import ReplayBuffer, SingleBuffer
from storages.n_step_buffer import NstepBuffer
from storages.priority_buffer import PrioritizedReplayBuffer
from storages.n_steper_buffer import NstepPrioritizedReplayBuffer
from storages.rollout_memory import MultiProcessMemory
from storages.tp_feeder import TPFeeder

def get_storage(state_space, act_space, args):
    if args.storage == 'replay':
        if args.buffer_type == 'vanilla':
            buffer = ReplayBuffer(args.rmsize)
        elif args.buffer_type == 'n_step':
            buffer = NstepBuffer(args.rmsize, n_step=args.n_step, gamma=args.discount)
        elif args.buffer_type == 'per':
            buffer = PrioritizedReplayBuffer(args.rmsize, args.alpha)
        elif args.buffer_type == 'n_stepper':
            buffer = NstepPrioritizedReplayBuffer(args.rmsize, args.alpha, n_step=args.n_step, gamma=args.discount)
        else:
            prRed("choose proper buffer type : vanilla | n_step | per | n_stepper")
            raise AttributeError
    elif args.storage == 'rollout':
        if args.buffer_type == 'vanilla':
            buffer = MultiProcessMemory(args.num_proc, state_space)
        else:
            prRed("choose proper rollout type : vanilla")
            raise AttributeError
    else:
        prRed("choose proper storage : replay | rollout")
        raise AttributeError

    storage = {'buffer': buffer}

    if args.storage == 'replay' and args.use_tp_graph:
        # loader = DataLoader(TPFeeder(data_path=args.tp_data_path, train=True),
        #                     batch_size=args.batch_size,
        #                     shuffle=True,
        #                     drop_last=True)
        # storage.update({'loader' : loader})
        tp_buffer = SingleBuffer(args.rmsize)
        storage.update({'tp_buffer' : tp_buffer})

    return storage