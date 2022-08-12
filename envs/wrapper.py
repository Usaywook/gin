import numpy as np
from gym import Wrapper, ObservationWrapper
from gym.spaces import Box, Dict

from utils.running_mean_std import RunningMeanStd
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv as _DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as _VecNormalize

class DummyEnv(Wrapper):
    def __init__(self, env):
        super(DummyEnv, self).__init__(env)
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class DummyVecEnv(_DummyVecEnv):
    def __init__(self, envs):
        super(DummyVecEnv, self).__init__(envs)
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def close(self):
        for env in self.envs:
            env.close()

    def get_tp_graph(self):
        tp_graph_features = []
        tp_adjacencies = []
        for env in self.envs:
            tp_graph_feature, tp_adjacency = env.tp_graph.node_n_adacency(env.tp_neighbors) # (C, T, V), (L+1, V, V)
            tp_graph_features.append(tp_graph_feature) # (N, C, T, V)
            tp_adjacencies.append(tp_adjacency) # (N, L + 1, V, V)
        return {'tp_graph_feature' : np.stack(tp_graph_features, axis=0), 'tp_adjacency' : np.stack(tp_adjacencies, axis=0)}


class VecNormalize(_VecNormalize):
    def __init__(self, venv, ob=True, ret=True, scale=10, clipob=100., cliprew=100., gamma=0.99, epsilon=1e-8):
        super(_VecNormalize, self).__init__(venv)
        self.dict_obs = isinstance(self.observation_space, Dict)
        if self.dict_obs:
            for k, v in self.observation_space.spaces.items():
                if k == 'feature':
                    obs_space_shape = v.shape
        else:
            obs_space_shape = self.observation_space.shape

        self.ob_rms = RunningMeanStd(shape=obs_space_shape) if ob else None
        self.clipob = clipob

        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.scale = scale

        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        if self.venv.training:
            rews = rews / self.scale
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs, update=True):
        if self.dict_obs:
            dict_obs = obs
            obs = obs['feature']
        if self.venv.training and self.ob_rms and update:
            self.ob_rms.update(obs)
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        if self.dict_obs: # apply norm obs
            dict_obs['feature'] = obs
        return dict_obs if self.dict_obs else obs

    def train(self):
        self.venv.train()

    def eval(self):
        self.venv.eval()

class RewardNormalize(Wrapper):
    def __init__(self, env, gamma=0.99, epsilon=1e-8, cliprew=10, scale=10, num_env=1):
        super(RewardNormalize, self).__init__(env)
        # self.ret_rms = RunningMeanStd(shape=())
        # self.num_env = num_env
        # self.ret = np.zeros(self.num_env)
        # self.gamma = gamma
        # self.epsilon = epsilon
        # self.cliprew = cliprew
        self.scale = scale

    def step(self, action, **kwargs):
        obs, rew, done, info = self.env.step(action, **kwargs)

        if self.training:
            # self.ret = self.ret * self.gamma + rew
            # self.ret_rms.update(self.ret)
            # rew = rew / np.sqrt(self.ret_rms.var + self.epsilon)
            # rew = np.clip(rew, -self.cliprew, self.cliprew)
            # rew = self.scale * rew
            # self.ret[done] = 0.
            rew = rew / self.scale

        return obs, rew, done, info

    def reset(self, **kwargs):
        # self.ret = np.zeros(self.num_env)
        return self.env.reset(**kwargs)

class ObservationNormalize(ObservationWrapper):
    def __init__(self, env, clipob=10, epsilon=1e-8):
        super(ObservationNormalize, self).__init__(env)
        self.dict_obs = isinstance(self.observation_space, Dict)
        if self.dict_obs:
            for k, v in self.observation_space.spaces.items():
                if k == 'feature':
                    obs_space_shape = v.shape
        else:
            obs_space_shape = self.observation_space.shape
        self.ob_rms = RunningMeanStd(shape=obs_space_shape)
        self.clipob = clipob
        self.epsilon = epsilon

    def observation(self, obs, update=True):
        if self.dict_obs:
            dict_obs = obs
            obs = obs['feature']
        if self.training and update:
            self.ob_rms.update(obs)
        obs = np.clip((obs - self.ob_rms.mean) /
                      np.sqrt(self.ob_rms.var + self.epsilon),
                      -self.clipob, self.clipob)
        return dict_obs if self.dict_obs else obs