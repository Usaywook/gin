from collections import deque
from storages.replaybuffer import ReplayBuffer

class NstepBuffer(ReplayBuffer):
    def __init__(self, size, n_step=1, gamma=0.99):
        super(NstepBuffer, self).__init__(size)
        self._n_step = n_step
        if self._n_step > 1:
            self._n_step_buffer = deque(maxlen=n_step)
            self._gamma = gamma

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._n_step > 1:
            data = self._get_n_step_info(data)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _get_n_step_info(self, data):
        self._n_step_buffer.append(data)

        obs_t, action, reward, obs_tp1, done = self._n_step_buffer[-1]

        for transition in reversed(list(self._n_step_buffer)[:-1]):
            _reward, _obs_tp1, _done = transition[-3:]

            reward = _reward + self._gamma * (1 - _done) * reward
            obs_tp1, done = (_obs_tp1, _done) if _done else (obs_tp1, done)

        return (obs_t, action, reward, obs_tp1, done)