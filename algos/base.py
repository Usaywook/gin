import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import gym
import wandb

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from utils.misc import prCyan
from utils import get_normalize_rms
from models.hmm import HMMEncoder

class Agent(ABC):
    def __init__(self, state_space, action_space, args):
        self.pixel = True if len(state_space.shape) > 1 else False
        self.discrete = True if isinstance(action_space, gym.spaces.Discrete) else False
        self.state_dim = state_space.shape[0]
        self.action_dim = action_space.n if self.discrete else action_space.shape[0]
        self.max_action = 1.0 if self.discrete else action_space.high[0]

        self.device = args.device
        self.seed = args.seed
        self.total_it = 0
        self.num_steps = args.num_steps

        self.start_timesteps = args.resume_step + args.start_timesteps if args.resume else args.start_timesteps
        self.batch_size = args.batch_size
        self.max_grad_norm = args.max_grad_norm
        self.discount = args.discount

        if args.mode != 'test':
            self.is_training = True

    @ abstractmethod
    def select_action(self, state):
        pass

    @ abstractmethod
    def update(self):
        pass

    @ abstractmethod
    def reset(self):
        pass

    @ abstractmethod
    def get_optimizer(self):
        pass

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def eval(self):
        for k, v in self._get_networks().items():
            v.eval()
        self.is_training = False

    def train(self):
        for k, v in self._get_networks().items():
            if not 'target' in k.split('_'):
                v.train()
        self.is_training = True

    def cuda(self):
        for k, v in self._get_networks().items():
            v.to(self.device)

    def _get_networks(self):
        networks = {}
        for k, v in self.__dict__.items():
            if isinstance(v, nn.Module) or isinstance(v, nn.Sequential):
                networks.update({k: v})
            elif isinstance(v, nn.ModuleList):
                for ind, mod in v:
                    if isinstance(mod, nn.Module) or isinstance(mod, nn.Sequential):
                        networks.update({k +'_{}'.format(str(ind)): mod})
            else:
                pass
        return networks

    def save_model(self, path, envs=None):
        if envs is not None:
            ob_rms = get_normalize_rms(envs)
            if ob_rms is not None:
                torch.save(ob_rms, "{}/ob_rms.pt".format(path))
        for k, v in self._get_networks().items():
            if isinstance(v, HMMEncoder):
                with open("{}/{}.pkl".format(path, k),"wb") as fw:
                    pickle.dump(v.model, fw)
            else:
                torch.save(v.state_dict(), "{}/{}.pt".format(path, k))
        prCyan('\tmodel is saved in {}'.format(path))

    def load_model(self, path, envs=None, save_check=None, resume_id=None, resume_step=0):
        if os.path.isfile("{}/ob_rms.pt".format(path)):
            ob_rms = torch.load("{}/ob_rms.pt".format(path), map_location=self.device)
            if envs is not None and ob_rms is not None:
                envs.ob_rms = ob_rms
                prCyan('\ob_rms is loaded from {}'.format(path))
        for k, v in self._get_networks().items():
            if isinstance(v, HMMEncoder):
                with open("{}/{}.pkl".format(path, k),"rb") as fr:
                    model = pickle.load(fr)
                    v.model = model
            else:
                v.load_state_dict(torch.load("{}/{}.pt".format(path, k), map_location=self.device))
        prCyan('\tmodel is loaded from {}'.format(path))

        if save_check is not None:
            save_log_path = '{}/save_log.txt'.format(path)
            with open(save_log_path, 'r') as f:
                header = f.readline().rstrip('\n').split(',')
            data = np.loadtxt(save_log_path, delimiter=',', skiprows=1)
            for i, k in enumerate(header):
                if i == 0:
                    if resume_id is not None:
                        self.total_it = resume_step # last iteration
                    else:
                        self.total_it = data[-1, i] if len(data.shape) > 1 else data[i]
                else:
                    save_check[k] = data[-1, i] if len(data.shape) > 1 else data[i]

    def set_writer(self, args):
        if not args.wandb:
            self.writer = SummaryWriter(args.output)
        else:
            run_name = args.algo
            if args.study is not None:
                run_name += '_{}'.format(args.study)
            if args.seed != -1:
                run_name += '_{}'.format(args.seed)
            if args.date:
                if args.resume:
                    run_name += '_{}'.format(args.resume_date)
                else:
                    run_name += '_{}'.format(datetime.today().strftime('%Y-%m-%d'))
            wandb.init(project=args.project, id=args.resume_id, resume=args.resume, name=run_name, config=args)
            wandb.watch(list(self._get_networks().values()), criterion=None, log="all", log_freq=args.validate_interval, log_graph=True)

    def write_summary(self, losses={}, train_logs={}, eval_logs={}, web=False):
        if web:
            wandb.log(train_logs, step=self.total_it)
            wandb.log(losses, step=self.total_it)
            wandb.log(eval_logs, step=self.total_it)
        else:
            for k, v in losses.items():
                self.writer.add_scalar(k, v, self.total_it)
            for k, v in train_logs.items():
                self.writer.add_scalar(k, v, self.total_it)
            for k, v in eval_logs.items():
                self.writer.add_scalar(k, v, self.total_it)
            for k, v in self._get_networks().items():
                for name, param in v.named_parameters():
                    self.writer.add_histogram('{}/'.format(k) + name, param.clone().cpu().data.numpy(), self.total_it)
