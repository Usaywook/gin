import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from gym.spaces import Dict
from algos.base import Agent
from layers.concatenator import Concatenator
from utils import soft_update, hard_update, prRed
from layers.build_model import build_cnn

from models.critic import TwinnedStateFunction, TwinnedStateActionFunction
from models.actor import CategoricalPolicy, GaussianPolicy, DeterministicPolicy

class SAC(Agent):
    def __init__(self, state_space, action_space, args):
        self.dict_obs = isinstance(state_space, Dict)
        super(SAC, self).__init__(state_space['feature'] if self.dict_obs else state_space,
                                  action_space, args)

        self._set_args(args)

        self._build_network(state_space, action_space, args)

        if args.mode == 'train':
            self._build_optimizer(args)

    def select_action(self, state):
        if self.dict_obs:
            kinetic_feature = torch.from_numpy(state['feature']).to(self.device).type(torch.float32) # (N, H1)
            dynamic_feature = torch.from_numpy(state['tracking']).to(self.device).type(torch.float32) # (N, V, H2)
            state = self.base(kinetic_feature, dynamic_feature) # (N, H)
        else:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if self.pixel:
            state = self.cnn(state)

        if self.is_training:
            action, _, _ = self.actor.sample(state)
        else:
            if self.discrete:
                action = self.actor.act(state)
            else:
                _, _, action = self.actor.sample(state)

        action = action.detach().cpu().data.numpy().flatten()
        if self.discrete:
            return action.item()
        else:
            return action

    def update(self, storage):
        self.total_it += 1
        if self.total_it < self.start_timesteps:
            return {}, {}
        if len(storage['buffer']) < self.batch_size:
            prRed("start_timesteps have to be greater than batch size")
            raise AssertionError

        if self.per:
            *mini_batch, weights, idxes = storage['buffer'].sample(self.batch_size, self.beta)
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            mini_batch = storage['buffer'].sample(self.batch_size)

        state, action, reward, mask, next_state = self._sample_from_batch(mini_batch)

        if self.per:
            critic_loss = self._update_critic(state, action, reward, mask, next_state, storage, weights, idxes)
        else:
            critic_loss = self._update_critic(state, action, reward, mask, next_state, storage)

        actor_loss, entropy = self._update_actor(state.detach())

        if self.automatic_entropy_tuning:
            alpha_loss = self._update_alpha(entropy)
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        if self.total_it % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        losses = {"actor_loss": actor_loss,
                  "alpha": self.alpha}

        losses.update(critic_loss)

        debug = {"entropy": entropy.item(),
                 "log_alpha": self.log_alpha.item()}

        if self.automatic_entropy_tuning:
            losses.update({"alpha_loss": alpha_loss})
            debug.update({"target_entropy": self.target_entropy})
        if self.per:
            debug.update({'beta' : self.beta})

        return losses, debug

    def _build_network(self, state_space, action_space, args):
        input_size = self.state_dim + args.hidden_size if self.dict_obs else self.state_dim
        if self.pixel:
            self.cnn = build_cnn(state_space.shape, args.cnn_hidden)
            input_size = args.cnn_hidden

        if not args.use_graph:
            self.base = Concatenator(input_dim=(self.state_dim, state_space['tracking'].shape[-1]),
                                     hidden_dim=args.hidden_size,
                                     activation=nn.Tanh())

        hidden_units = (args.hidden_size, args.hidden_size)
        if self.discrete:
            self.critic = TwinnedStateFunction(input_size, self.action_dim, hidden_units)
            self.critic_target = copy.deepcopy(self.critic)
        else:
            self.critic = TwinnedStateActionFunction(input_size, self.action_dim, hidden_units)
            self.critic_target = copy.deepcopy(self.critic)

        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Stochastic":
            if self.discrete:
                self.actor = CategoricalPolicy(input_size, self.action_dim, args.hidden_size).to(self.device)
            else:
                self.actor = GaussianPolicy(input_size, self.action_dim, args.hidden_size, action_space).to(self.device)

            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                if self.discrete:
                    self.target_entropy = -torch.log(
                        torch.tensor(1.0 / self.action_dim).to(self.device)).item() * args.entropy_sensitivity
                else:
                    self.target_entropy = - torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha = F.softplus(self.log_alpha).clamp(min=1e-8, max=1e+8).item()
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicPolicy(input_size, self.action_dim, args.hidden_size, action_space).to(self.device)

    def _build_optimizer(self, args):
        parameters = list(self.critic.parameters())
        if self.dict_obs:
            parameters += list(self.base.parameters())
        if self.pixel:
            parameters += list(self.cnn.parameters())
        self.critic_optim = Adam(parameters, lr=args.lr)

        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

    def _set_args(self, args):
        self.tau = args.tau
        self.alpha = args.temperature
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.per = (args.buffer_type == 'per' or args.buffer_type == 'n_stepper')
        if self.per:
            self._set_per_args(args)

    def _set_per_args(self, args):
        self.beta = args.beta
        self.e = args.e
        self.beta_increment_per_sampling = args.beta_increment_per_sampling

    def _sample_from_batch(self, mini_batch):
        if self.dict_obs:
            kinetic_feature = torch.from_numpy(np.vstack(list(map(lambda x: x.item()['feature'], mini_batch[0])))).type(torch.float32).to(self.device)
            dynamic_feature = torch.from_numpy(np.vstack(list(map(lambda x: x.item()['tracking'], mini_batch[0])))).type(torch.float32).to(self.device)
            next_kinetic_feature = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['feature'], mini_batch[3])))).type(torch.float32).to(self.device)
            next_dynamic_feature = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['tracking'], mini_batch[3])))).type(torch.float32).to(self.device)
            state = self.base(kinetic_feature, dynamic_feature)
            with torch.no_grad():
                next_state = self.base(next_kinetic_feature, next_dynamic_feature)
        else:
            state = torch.FloatTensor(np.vstack(mini_batch[0])).to(self.device)
            next_state = torch.FloatTensor(np.vstack(mini_batch[3])).to(self.device)

        action = torch.FloatTensor(np.vstack(mini_batch[1])).to(self.device)
        reward = torch.FloatTensor(np.vstack(mini_batch[2])).to(self.device)
        mask = torch.FloatTensor(1 - np.vstack(mini_batch[4])).to(self.device)

        if self.pixel:
            state = self.cnn(state)
            next_state = self.cnn(next_state)

        return state, action, reward, mask, next_state

    def _update_critic(self, state, action, reward, mask, next_state, storage, weights=None, idxes=None):
        # target_q
        with torch.no_grad():
            if self.discrete:
                _, next_pi_probs, next_log_pi = self.actor.sample(next_state)
                qf1_next_target, qf2_next_target = self.critic_target(next_state)
                min_qf_next_target = next_pi_probs * (torch.min(qf1_next_target,
                                                                qf2_next_target) - self.alpha * next_log_pi)  # expectation with action probability
                min_qf_next_target = min_qf_next_target.sum(dim=-1, keepdim=True)
            else:
                next_action, next_log_pi, _ = self.actor.sample(next_state)
                qf1_next_target, qf2_next_target = self.critic_target(next_state, next_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
            assert reward.shape == min_qf_next_target.shape
            next_q_value = reward + mask * self.discount * (min_qf_next_target)

        # current_q
        if self.discrete:
            qf1, qf2 = self.critic(state)
            qf1 = qf1.gather(1, action.long())
            qf2 = qf2.gather(1, action.long())
        else:
            qf1, qf2 = self.critic(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step

        qf1_error = next_q_value - qf1
        qf2_error = next_q_value - qf2

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = 0.5 * (qf1_error ** 2).mean()
        qf2_loss = 0.5 * (qf2_error ** 2).mean()
        qf_loss = qf1_loss + qf2_loss

        if self.per:
            qf_loss = (weights * qf_loss).mean()
            # update priority
            priority = 0.5 * torch.abs(qf1_error) + 0.5 * torch.abs(qf2_error)
            errors = priority.detach().cpu().numpy().flatten()
            storage['buffer'].update_priorities(idxes, np.clip(errors, self.e, 1000))

        self.critic_optim.zero_grad()
        nn.utils.clip_grad_norm_(
            list(itertools.chain(*[g['params'] for g in self.critic_optim.param_groups])),
            self.max_grad_norm)
        qf_loss.backward()
        self.critic_optim.step()

        return {'q1_loss': qf1_loss.item(),
                'q2_loss': qf2_loss.item()}

    def _update_actor(self, state):
        if self.discrete:
            _, pi_probs, log_pi = self.actor.sample(state)
            qf1_pi, qf2_pi = self.critic(state)
        else:
            pi, log_pi, _ = self.actor.sample(state)
            qf1_pi, qf2_pi = self.critic(state, pi)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        if self.discrete:
            inside_term = self.alpha * log_pi - min_qf_pi
            actor_loss = torch.sum(pi_probs * inside_term, dim=-1).mean()
        else:
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        if self.discrete:
            log_pi = torch.sum(pi_probs * log_pi, dim=-1)
        entropy = - log_pi.mean()

        return actor_loss.item(), entropy

    def _update_alpha(self, entropy):
        alpha = F.softplus(self.log_alpha).clamp(min=1e-8, max=1e+8)
        alpha_loss = - alpha * (self.target_entropy - entropy.detach())

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
        self.alpha_optim.step()

        self.alpha = alpha.item()

        return alpha_loss.item()

    def reset(self):
        pass

    def get_optimizer(self):
        optimizer = {"critic": self.critic_optim,
                     "actor": self.actor_optim}
        if self.automatic_entropy_tuning:
            optimizer.update({"alpha": self.alpha_optim})
        return optimizer

