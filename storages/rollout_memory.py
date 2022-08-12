from itertools import chain
import numpy as np
import torch
from gym.spaces import Dict

class Trajectory(object):
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.log_probs = []
        self.dones = []
        self.done = False

    def __len__(self):
        return len(self.observations)

class SingleProcessMemory(object):
    def __init__(self, state_space):
        self.dict_obs = isinstance(state_space, Dict)
        self.trajectories = []

    def reset(self):
        self.trajectories = []
        self.trajectory = Trajectory()

    def add(self, state, action, reward, done, cost, log_prob):
        self.trajectory.observations.append(state)
        self.trajectory.actions.append(action)
        self.trajectory.rewards.append(reward)
        self.trajectory.costs.append(cost)
        self.trajectory.log_probs.append(log_prob)
        self.trajectory.done = done
        if self.trajectory.done:
            self.trajectories.append(self.trajectory)
            self.trajectory = Trajectory()

    def sample(self):
        observations = list(chain(*[traj.observations for traj in self.trajectories]))
        actions = np.vstack(list(chain(*[traj.actions for traj in self.trajectories])))
        rewards = np.stack(list(chain(*[traj.rewards for traj in self.trajectories])))
        costs = np.stack(list(chain(*[traj.costs for traj in self.trajectories])))
        log_probs = np.stack(list(chain(*[traj.log_probs for traj in self.trajectories])))
        if self.dict_obs:
            path_features = np.vstack(list(map(lambda x: x['feature'], observations)))
            dynamic_features = np.stack(list(map(lambda x: x['tracking'], observations)))
            return path_features, dynamic_features, actions, rewards, costs, log_probs
        else:
            path_features = np.vstack(observations)
            return path_features, actions, rewards, costs, log_probs

    def get_size(self):
        return np.array([len(traj) for traj in self.trajectories])

    def __len__(self):
        return np.sum(self.get_size()).astype(int)

class MultiProcessMemory(object):
    def __init__(self, num_process, state_space):
        self.num_process = num_process
        self.dict_obs = isinstance(state_space, Dict)

    def reset(self):
        self.trajectories = np.asarray([Trajectory() for i in range(self.num_process)])

    def sample(self):
        observations = np.vstack(list(chain(*[traj.observations for traj in self.trajectories])))
        actions = np.vstack(list(chain(*[traj.actions for traj in self.trajectories])))
        rewards = np.vstack(list(chain(*[traj.rewards for traj in self.trajectories])))
        costs = np.vstack(list(chain(*[traj.costs for traj in self.trajectories])))
        log_probs = np.vstack(list(chain(*[traj.log_probs for traj in self.trajectories])))
        dones = np.vstack(list(chain(*[traj.dones for traj in self.trajectories])))

        if self.dict_obs:
            states = [np.stack(list(map(lambda x: x.item()[key], observations)), axis=0) for key in observations[0].item().keys()]
            return *states, actions, rewards, costs, log_probs, dones
        else:
            observations = np.vstack(observations)
            return observations, actions, rewards, costs, log_probs, dones

    def add(self, state, action, reward, done, cost, log_prob=None):
        for ind in range(self.num_process):
            if self.dict_obs:
                self.trajectories[ind].observations.append({k : v[ind] for k, v in state.items()})
            else:
                self.trajectories[ind].observations.append(state[ind])
            self.trajectories[ind].actions.append(action[ind])
            self.trajectories[ind].rewards.append(reward[ind])
            self.trajectories[ind].costs.append(cost[ind])
            if log_prob is not None:
                self.trajectories[ind].log_probs.append(log_prob[ind])
            self.trajectories[ind].dones.append(done[ind])

    def get_size(self):
        traj_size = np.array([len(traj) for traj in self.trajectories])
        dones = np.stack(list(chain(*[traj.dones for traj in self.trajectories])))
        traj_limit = np.arange(traj_size.sum())[dones] + 1.
        step_size = np.concatenate([traj_limit[0:1], traj_limit[1:] - traj_limit[:-1]])
        return step_size

    def get_summary(self):
        dones = np.stack(list(chain(*[traj.dones for traj in self.trajectories])))
        rewards = np.stack(list(chain(*[traj.rewards for traj in self.trajectories])))
        costs = np.stack(list(chain(*[traj.costs for traj in self.trajectories])))
        traj_limit = np.arange(len(self.trajectories[0]) * self.num_process)
        traj_limit = traj_limit[dones] + 1
        step_sizes = np.concatenate([[traj_limit[0]], traj_limit[1:] - traj_limit[:-1]])
        episode_returns = np.mean(list(map(np.sum, np.split(rewards, traj_limit[:-1], axis=0))))
        episode_costs = np.mean(list(map(np.sum,np.split(costs, traj_limit[:-1], axis=0))))
        episode_steps = np.mean(step_sizes)
        summary = {'num_episode' : len(step_sizes),
                   'train_return': episode_returns,
                   'train_cost': episode_costs,
                   'train_step': episode_steps}
        return summary

    def __getitem__(self, i):
        return self.trajectories[i] # i th process trajectory

    def __len__(self):
        return np.sum(self.get_size()).astype(int)
