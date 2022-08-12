import torch
from gym.spaces import Dict
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    '''
    For Current Policy Buffer
    '''
    def __init__(self, state_space, action_space, args):
        self.dict_obs = isinstance(state_space, Dict)
        state_size = state_space.shape
        self.action_shape = 1 if action_space.__class__.__name__ == 'Discrete' else action_space.shape[0]
        self.action_type = torch.long if action_space.__class__.__name__ == 'Discrete' else torch.float

        self.states = torch.empty((args.rollout_length, *state_size), dtype=torch.float, device=args.device)
        self.actions = torch.empty((args.rollout_length, self.action_shape), dtype=self.action_type, device=args.device)
        self.rewards = torch.empty((args.rollout_length, 1), dtype=torch.float, device=args.device)
        self.dones = torch.empty((args.rollout_length, 1), dtype=torch.float, device=args.device)
        self.log_probs = torch.empty((args.rollout_length, 1), dtype=torch.float, device=args.device)

        self.rollout_length = args.rollout_length
        self.device = args.device
        self._step = 0
        self._n = 0

    def add(self, state, action, reward, done, log_prob):
        self.states[self._step].copy_(torch.from_numpy(state))
        self.actions[self._step].copy_(torch.FloatTensor([action]).flatten())
        self.rewards[self._step].copy_(torch.FloatTensor([reward]))
        self.dones[self._step].copy_(torch.FloatTensor([done]))
        self.log_probs[self._step].copy_(log_prob)

        self._step = (self._step + 1) % self.rollout_length
        self._n = min(self._n + 1, self.rollout_length)

    def get(self):
        assert self._step % self.rollout_length == 0
        start = (self._step - self.rollout_length) % self.rollout_length
        idxes = slice(start, start + self.rollout_length)
        return (self.states[idxes],
                self.actions[idxes],
                self.rewards[idxes],
                self.dones[idxes],
                self.log_probs[idxes])

    def feed_forward_generator(self, batch_size):
        sampler = BatchSampler(SubsetRandomSampler(range(self.rollout_length)),
                                    batch_size,
                                    drop_last=True)
        for indices in sampler:
            yield (self.states[indices],
                   self.actions[indices],
                   self.rewards[indices],
                   self.dones[indices],
                   self.log_probs[indices])

    def __len__(self):
        return self._n