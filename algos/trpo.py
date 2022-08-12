import itertools
import time

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from gym.spaces import Dict

from models.actor import GaussianPolicy
from models.critic import StateValueFunction
from utils.misc import flat_grad, flat_hessian, get_flat_params, set_params
from algos.base import Agent

class TRPO(Agent):
    def __init__(self, state_space, action_space, args):
        self.dict_obs = isinstance(state_space, Dict)
        super(TRPO, self).__init__(state_space['feature'] if self.dict_obs else state_space,
                                   action_space, args)
        input_size = self.state_dim + args.hidden_size if self.dict_obs else self.state_dim

        self.actor = GaussianPolicy(input_size, self.action_dim, args.hidden_size, action_space)
        hidden_units = (args.hidden_size, args.hidden_size)
        self.critic = StateValueFunction(input_size, hidden_units=hidden_units)

        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)
        if self.dict_obs:
            if 'tracking' in state_space.spaces.keys():
                self.linear = nn.Linear(state_space['tracking'].shape[-1], args.hidden_size)
                self.linear_activation = nn.Tanh()
                torch.nn.init.xavier_uniform_(self.linear.weight, gain=1)
                self.critic_optim.add_param_group({'params':self.linear.parameters(), 'lr':args.linear_lr})

        self.lambd = args.lambd
        self.l2_reg = args.l2_reg
        self.max_kl = args.max_kl
        self.damping = args.damping
        self.n_epoch = args.n_epoch
        self.num_proc = args.num_proc

    def select_action(self, state):
        if self.dict_obs:
            path_feature = torch.from_numpy(state['feature']).to(self.device).type(torch.float32) # (N, H1)
            dynamic_feature = torch.from_numpy(state['tracking']).to(self.device).type(torch.float32) # (N, V, C)
            dynamic_feature = self.linear(dynamic_feature) # (N, V, H2)
            dynamic_feature = self.linear_activation(dynamic_feature)
            dynamic_feature = torch.max(dynamic_feature, dim=1)[0]  # (N, H2)
            state = torch.cat([path_feature, dynamic_feature], dim=-1) # (N, H1 + H2)
        else:
            state = torch.FloatTensor(state).to(self.device)

        action, log_prob, mean = self.actor.sample(state)

        if self.is_training:
            log_prob = log_prob.detach().cpu().data.numpy()
            action = action.detach().cpu().data.numpy()
            return action, log_prob
        else:
            mean = mean.detach().cpu().data.numpy()
            return mean

    def update(self, storage):
        N = len(storage['buffer'])
        self.total_it += int(N / self.num_proc)
        start_time = time.time()

        mini_batch = storage['buffer'].sample()
        if self.dict_obs:
            path_feature = torch.from_numpy(mini_batch[0]).type(torch.float32).to(self.device)
            dynamic_feature = torch.from_numpy(mini_batch[1]).type(torch.float32).to(self.device)
            actions = torch.from_numpy(mini_batch[2]).type(torch.float32).to(self.device)
            rewards = torch.from_numpy(mini_batch[3]).type(torch.float32).to(self.device)
            fixed_log_prob = torch.from_numpy(mini_batch[5]).type(torch.float32).to(self.device)
            dones = torch.from_numpy(mini_batch[6]).type(torch.float32).to(self.device)
        else:
            states = torch.from_numpy(mini_batch[0]).type(torch.float32).to(self.device)
            actions = torch.from_numpy(mini_batch[1]).type(torch.float32).to(self.device)
            rewards = torch.from_numpy(mini_batch[2]).type(torch.float32).to(self.device)
            fixed_log_prob = torch.from_numpy(mini_batch[4]).type(torch.float32).to(self.device)
            dones = torch.from_numpy(mini_batch[5]).type(torch.float32).to(self.device)

        for epoch in range(self.n_epoch):
            with torch.no_grad():
                if self.dict_obs:
                    global_feature = self.linear(dynamic_feature)
                    global_feature = self.linear_activation(global_feature)
                    global_feature = torch.max(global_feature, dim=1)[0]
                    states = torch.cat([path_feature, global_feature], dim=-1)
                values = self.critic(states)

            # step 1: get returns and GAEs
            targets, gaes = self._get_ret_gae(rewards, (1.0 - dones), values, self.discount, self.lambd)

            # step 2: train critic several steps with respect to returns
            states_train = (path_feature, dynamic_feature) if self.dict_obs else states
            critic_loss = self._update_nn_regressor(self.critic, self.critic_optim, states_train, targets, 5)

            debugs = self._update_policy(states, actions, fixed_log_prob, gaes)

        debugs.update({'elapsed_time': time.time() - start_time})

        return {"critic_loss" : critic_loss.item()}, debugs

    def _update_policy(self, states, actions, fixed_log_prob, gaes):
        # step 3: get gradient of surrogate loss and hessian of kl
        sur_loss, mu, log_std = self._sur_loss(states, actions, gaes, fixed_log_prob, True)
        loss_grad = flat_grad(sur_loss, self.actor.parameters())

        # step 4: get step direction through conjugate gradient
        stepdir = self._conjugate_gradients(states, loss_grad.data, 10)

        # step 5: get lagrange_mutiplier and full step (step_size = 1 / lagrange_mutiplier)
        shs = 0.5 * (stepdir * self._fisher_vector_product(states, stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)[0]
        fullstep = stepdir / lm
        debugs = {"lagrange multiplier": lm.item(), "grad_norm": loss_grad.norm(), "surrogate_loss": sur_loss.item()}

        # step 5: do backtracking line search for n times
        debug = self._linesearch(sur_loss, mu, log_std,
                                    loss_grad, fullstep,
                                    states, actions, gaes, fixed_log_prob)
        debugs.update(debug)
        return debugs

    def _get_ret_gae(self, rewards, masks, values, discount, lambd):
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)

        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + discount * running_returns * masks[t]
            running_tderror = rewards[t] + discount * previous_value * masks[t] - values.data[t]
            running_advants = running_tderror + discount * lambd * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants

        advants = (advants - advants.mean()) / (advants.std() + 1e-6)
        return returns, advants

    def _sur_loss(self, states, actions, advantages, fixed_log_prob, mulogstd=False):
        if mulogstd:
            log_prob, mu, log_std = self.actor.log_prob(states, actions, mulogstd)
        else:
            log_prob = self.actor.log_prob(states, actions)

        sur_loss = advantages * torch.exp(log_prob - fixed_log_prob)

        if mulogstd:
            return sur_loss.mean(), mu, log_std
        else:
            return sur_loss.mean()

    def _kl_divergence(self, mu, log_std, mu_old, log_std_old):
        std = log_std.exp()
        std_old = log_std_old.exp()

        mu_old = mu_old.detach()
        log_std_old = log_std_old.detach()
        std_old = std_old.detach()

        kl = log_std - log_std_old + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def _fisher_vector_product(self, states, v):
        mu, log_std = self.actor(states)
        kl = self._kl_divergence(mu, log_std, mu, log_std)
        kl = kl.mean()

        flat_grad_kl = flat_grad(kl, self.actor.parameters(), create_graph=True)

        kl_v = (flat_grad_kl * v).sum()
        hessian_kl = flat_hessian(kl_v, self.actor.parameters())

        return hessian_kl + v * self.damping

    def _conjugate_gradients(self, states, b, nsteps, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self._fisher_vector_product(states, p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def _linesearch(self, sur_loss_old, mu_old, log_std_old,
                    loss_grad, fullstep,
                    states, actions, gaes, fixed_log_prob,
                    max_backtracks=10, accept_ratio=.5):
        prev_params = get_flat_params(self.actor)

        flag = False
        expected_improve = (loss_grad * fullstep).sum(0, keepdim=True)
        for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            new_params = prev_params + stepfrac * fullstep
            set_params(self.actor, new_params)

            with torch.no_grad():
                sur_loss, mu, log_std = self._sur_loss(states, actions, gaes, fixed_log_prob, True)
                mu, log_std = self.actor(states)
                kl = self._kl_divergence(mu, log_std, mu_old, log_std_old).mean()

                actual_improve = sur_loss - sur_loss_old
                expected_improve *= stepfrac
                ratio = actual_improve / expected_improve

            if ratio > accept_ratio and actual_improve > 0 and kl < self.max_kl:
                flag = True
                break

        if not flag:
            set_params(self.actor, prev_params)

        debug = {"policy_update" : flag,
                 "backtrack_step" : _n_backtracks,
                 "sur_loss_before" : sur_loss_old.item(),
                 "sur_loss_after" : sur_loss.item(),
                 "act_improve": actual_improve.item(),
                 "exp_improve": expected_improve.item(),
                 "imp_ratio" : ratio.item(),
                 "kld" : kl.item()}

        return debug

    def _update_nn_regressor(self, nn_regressor, optimizer, states, targets, n_iters=1):
        if self.dict_obs:
            path_feature, dynamic_feature = states
            N = path_feature.shape[0]
        else:
            N = states.shape[0]

        total_loss = []
        for _ in range(n_iters):
            total_ind = np.arange(N)
            np.random.shuffle(total_ind)
            for j in range(N // self.batch_size):
                mini_batch_ind = total_ind[self.batch_size * j : self.batch_size * (j + 1)]
                if self.dict_obs:
                    global_feature = self.linear(dynamic_feature)
                    global_feature = self.linear_activation(global_feature)
                    global_feature = torch.max(global_feature, dim=1)[0]
                    states = torch.cat([path_feature, global_feature], dim=-1)

                predictions = nn_regressor(states[mini_batch_ind])
                loss = (predictions - targets[mini_batch_ind]).pow(2).mean()

                flat_params = get_flat_params(nn_regressor)
                if self.dict_obs:
                    flat_params = torch.cat([flat_params, get_flat_params(self.linear)])
                l2_loss = self.l2_reg * torch.sum(torch.pow(flat_params, 2))
                loss += l2_loss
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(
                    list(itertools.chain(*[g['params'] for g in optimizer.param_groups])),
                    self.max_grad_norm)
                optimizer.step()

        return np.mean(total_loss)

    def reset(self):
        pass

    def get_optimizer(self):
        return {"critic": self.critic_optim}
