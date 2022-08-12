import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from gym.spaces import Dict
from scipy.stats import norm
from algos.sac import SAC
from utils import soft_update, hard_update, prRed

from models.critic import StateActionFunction


class WCSAC(SAC):
    def __init__(self, state_space, action_space, args):
        self.dict_obs = isinstance(state_space, Dict)
        super(SAC, self).__init__(state_space['feature'] if self.dict_obs else state_space,
                                  action_space, args)

        self._set_args(args)

        self._build_network(state_space, action_space, args)

        if args.mode == 'train':
            self._build_optimizer(args)

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

        state, action, reward, mask, next_state, cost = self._sample_from_batch(mini_batch)

        if self.per:
            critic_loss, cvar = self._update_critic(state, action, reward, mask, next_state, cost, storage, weights, idxes)
        else:
            critic_loss, cvar = self._update_critic(state, action, reward, mask, next_state, cost, storage)

        actor_loss, entropy = self._update_actor(state.detach(), cvar)

        if self.automatic_entropy_tuning:
            alpha_loss = self._update_alpha(entropy)
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        kappa_loss = self._update_kappa(cvar)

        if self.total_it % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.cost_target, self.cost, self.tau)
            soft_update(self.cost_var_target, self.cost_var, self.tau)

        losses = {"actor_loss": actor_loss,
                  "kappa_loss": kappa_loss,
                  "alpha": self.alpha,
                  "kappa": self.kappa}

        losses.update(critic_loss)

        debug = {"entropy": entropy.item(),
                 "log_alpha": self.log_alpha.item(),
                 "log_kappa": self.log_kappa.item(),
                 "cost_const": self.cost_constraint,
                 "pdf_cdf": self.pdf_cdf,
                 "cvar": cvar.item()}

        if self.automatic_entropy_tuning:
            losses.update({"alpha_loss": alpha_loss})
            debug.update({"target_entropy": self.target_entropy})
        if self.per:
            debug.update({'beta' : self.beta})

        return losses, debug

    def _set_args(self, args):
        super(WCSAC, self)._set_args(args)

        self.verbose = args.verbose
        self.pdf_cdf = args.cvar**(-1) * norm.pdf(norm.ppf(args.cvar))
        # max_traj_len = args.max_episode_steps / args.skip
        self.cost_constraint = args.max_constraint # * (1 - self.discount ** max_traj_len) / (1 - self.discount) / max_traj_len # expected sum of discounted cost_lim
        self.damp_scale = args.damp_scale

    def _build_network(self, state_space, action_space, args):
        super(WCSAC, self)._build_network(state_space, action_space, args)
        input_size = self.state_dim + args.hidden_size if self.dict_obs else self.state_dim
        hidden_units = (args.hidden_size, args.hidden_size)

        self.cost = StateActionFunction(input_size, self.action_dim, hidden_units=hidden_units, output_activation=nn.ReLU(inplace=False))
        self.cost_target = copy.deepcopy(self.cost)

        self.cost_var = StateActionFunction(input_size, self.action_dim, hidden_units=hidden_units, output_activation=nn.Softplus())
        self.cost_var_target = copy.deepcopy(self.cost_var)

        hard_update(self.cost_target, self.cost)
        hard_update(self.cost_var_target, self.cost_var)

        self.log_kappa = torch.zeros(1, requires_grad=True, device=self.device)
        self.kappa = F.softplus(self.log_kappa).clamp(min=1e-8, max=1e+8).item()

    def _build_optimizer(self, args):
        parameters = list(self.critic.parameters())
        parameters += list(self.cost.parameters())
        parameters += list(self.cost_var.parameters())
        if self.dict_obs:
            parameters += list(self.base.parameters())
        self.critic_optim = Adam(parameters, lr=args.lr)

        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.kappa_optim = Adam([self.log_kappa], lr=args.lr)

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
        cost = torch.FloatTensor(np.vstack(mini_batch[5])).to(self.device)

        if self.pixel:
            state = self.cnn(state)
            next_state = self.cnn(next_state)

        return state, action, reward, mask, next_state, cost

    def _update_critic(self, state, action, reward, mask, next_state, cost, storage, weights=None, idxes=None):
        # current_q, current_qc, current_qc_var
        qf1, qf2 = self.critic(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qc = self.cost(state, action)
        qc_var = self.cost_var(state, action).clamp(min=1e-8, max=1e+8)

        # target_q
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
            qc_next_target = self.cost_target(next_state, next_action)
            qc_var_next_target = self.cost_var_target(next_state, next_action).clamp(min=1e-8, max=1e+8)

            assert reward.shape == min_qf_next_target.shape
            next_q_value = reward + mask * self.discount * min_qf_next_target
            assert cost.shape == qc_next_target.shape
            next_qc_value = cost + mask * self.discount * qc_next_target
            next_qc_var_value = cost ** 2  + 2 * self.discount * cost * qc_next_target + \
                self.discount ** 2 * qc_var_next_target + self.discount ** 2 * qc_next_target ** 2 - qc ** 2
            next_qc_var_value = next_qc_var_value.clamp(min=1e-8, max=1e+8)

        qf1_error = next_q_value - qf1
        qf2_error = next_q_value - qf2
        qc_error = next_qc_value - qc
        qc_var_error = qc_var + next_qc_var_value - 2 * ((qc_var * next_qc_var_value) ** 0.5)

        qf1_loss = 0.5 * (qf1_error ** 2).mean()
        qf2_loss = 0.5 * (qf2_error ** 2).mean()
        qc_loss = 0.5 * (qc_error ** 2).mean()
        qc_var_loss = 0.5 * qc_var_error.mean()
        qf_loss = qf1_loss + qf2_loss + qc_loss + qc_var_loss

        cvar = qc + self.pdf_cdf * torch.sqrt(qc_var)

        if self.verbose:
            print(f'qf1: {qf1_loss.item()}, qf2: {qf2_loss.item()}, qc: {qc_loss.item()}, qc_var: {qc_var_loss.item()}')

        if self.per:
            qf_loss = (weights * qf_loss).mean()
            # update priority
            reward_priority = 0.5 * torch.abs(qf1_error) + 0.5 * torch.abs(qf2_error)
            cost_priority = 0.5 * torch.abs(qc_error) + 0.5 * torch.abs(qc_var_error)
            priority = (reward_priority + cost_priority).detach().cpu().numpy().flatten()
            storage['buffer'].update_priorities(idxes, np.clip(priority, self.e, 1000))

        self.critic_optim.zero_grad()
        nn.utils.clip_grad_norm_(
            list(itertools.chain(*[g['params'] for g in self.critic_optim.param_groups])),
            self.max_grad_norm)
        qf_loss.backward()
        self.critic_optim.step()

        return {'q1_loss': qf1_loss.item(),
                'q2_loss': qf2_loss.item(),
                'qc_loss': qc_loss.item(),
                'qc_var_loss': qc_var_loss.item()}, cvar.mean().detach()

    def _update_actor(self, state, cvar):
        pi, log_pi, _ = self.actor.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        qc_pi = self.cost(state, pi)
        qc_pi_var = self.cost_var(state, pi).clamp(min=1e-8, max=1e+8)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        damp = self.damp_scale * (self.cost_constraint - cvar)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi + (self.kappa - damp) * (qc_pi + self.pdf_cdf * (qc_pi_var ** 0.5))).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        entropy = - log_pi.mean()

        return actor_loss.item(), entropy

    def _update_kappa(self, cvar):
        kappa = F.softplus(self.log_kappa).clamp(min=1e-8, max=1e+8)
        kappa_loss = kappa * (self.cost_constraint - cvar) # increase kappa if loss < 0 else decrease kappa
        self.kappa_optim.zero_grad()
        kappa_loss.backward()
        nn.utils.clip_grad_norm_([self.log_kappa], self.max_grad_norm)
        self.kappa_optim.step()

        self.kappa = kappa.item()

        return kappa_loss.item()

    def reset(self):
        pass

    def get_optimizer(self):
        optimizer = super(WCSAC, self).get_optimizer()
        optimizer.update({"kappa": self.kappa_optim})
        return optimizer
