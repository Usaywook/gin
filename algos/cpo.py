import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from algos.trpo import TRPO
from models.critic import StateValueFunction
from utils.misc import prRed, flat_grad, get_flat_params, set_params


class CPO(TRPO):
    def __init__(self, state_space, action_space, args):
        super(CPO, self).__init__(state_space, action_space, args)
        input_size = self.state_dim + args.hidden_size if self.dict_obs else self.state_dim

        hidden_units = (args.hidden_size, args.hidden_size)
        self.cost = StateValueFunction(input_size, hidden_units=hidden_units)

        self.cost_optim = Adam(self.cost.parameters(), lr=args.cost_lr)
        # if self.dict_obs:
        #     if 'tracking' in state_space.spaces.keys():
        #         self.cost_optim.add_param_group({'params':self.linear.parameters(), 'lr':args.linear_lr})

        self.n_epoch = args.n_epoch
        self.val_iter = args.val_iter
        self.cost_iter = args.cost_iter
        self.cg_max_iter = args.cg_max_iter
        self.line_search_max_iter = args.line_search_max_iter

        self.max_constraint = args.max_constraint
        self.accept_ratio = args.accept_ratio
        self.damping = args.damping
        self.lambd_c = args.lambd_c
        self.line_search_coef = args.line_search_coef
        self.verbose = args.verbose
        self.fusion = args.fusion

        if self.fusion:
            self.state_prev = None
            self.disc_reward_prev = None
            self.disc_cost_prev = None

    def update(self, storage):
        N = len(storage['buffer'])
        self.total_it += int(N / self.num_proc)

        if N < self.batch_size:
            prRed('rollout length have to be greater than batch size')
            raise AssertionError
        mini_batch = storage['buffer'].sample()

        if self.dict_obs:
            path_feature = torch.from_numpy(mini_batch[0]).type(torch.float32).to(self.device)
            dynamic_feature = torch.from_numpy(mini_batch[1]).type(torch.float32).to(self.device)
            action = torch.from_numpy(mini_batch[2]).type(torch.float32).to(self.device)
            reward = torch.from_numpy(mini_batch[3]).type(torch.float32).to(self.device)
            cost = torch.from_numpy(mini_batch[4]).type(torch.float32).to(self.device)
            log_prob = torch.from_numpy(mini_batch[5]).type(torch.float32).to(self.device)
            done = mini_batch[6]

        else:
            state = torch.from_numpy(mini_batch[0]).type(torch.float32).to(self.device)
            action = torch.from_numpy(mini_batch[1]).type(torch.float32).to(self.device)
            reward = torch.from_numpy(mini_batch[2]).type(torch.float32).to(self.device)
            cost = torch.from_numpy(mini_batch[3]).type(torch.float32).to(self.device)
            log_prob = torch.from_numpy(mini_batch[4]).type(torch.float32).to(self.device)
            done = mini_batch[5]

        start_indices = np.concatenate([[0], np.where(done)[0][:-1] + 1])
        done = torch.from_numpy(done).type(torch.float32).to(self.device)

        start_time = time.time()
        for epoch in range(self.n_epoch):
            with torch.no_grad():
                if self.dict_obs:
                    global_feature = self.linear(dynamic_feature)
                    global_feature = self.linear_activation(global_feature)
                    global_feature = torch.max(global_feature, dim=1)[0]
                    state = torch.cat([path_feature, global_feature], dim=-1)
                pred_value = self.critic(state)
                pred_cost = self.cost(state)

            # step 1: get returns, c-returns and advanges, c-advanges
            disc_reward, reward_adv = self._get_ret_gae(reward, 1 - done, pred_value, self.discount, self.lambd)
            disc_cost, cost_adv = self._get_ret_gae(cost, 1 - done, pred_cost, self.discount, self.lambd_c)

            if self.fusion:
                if self.state_prev is not None:
                    if self.dict_obs:
                        state_train = tuple(map(lambda x : torch.cat([*x]), zip((path_feature,
                                                                                dynamic_feature),
                                                                                self.state_prev)))
                    else:
                        state_train = torch.cat([state, self.state_prev])
                    disc_reward_train = torch.cat([disc_reward, self.disc_reward_prev])
                    disc_cost_train = torch.cat([disc_cost, self.disc_cost_prev])
                else:
                    state_train = (path_feature, dynamic_feature) if self.dict_obs else state
                    disc_reward_train = disc_reward
                    disc_cost_train = disc_cost

                self.state_prev = (path_feature, dynamic_feature) if self.dict_obs else state
                self.disc_reward_prev = disc_reward
                self.disc_cost_prev = disc_cost
            else:
                state_train = (path_feature, dynamic_feature) if self.dict_obs else state
                disc_reward_train = disc_reward
                disc_cost_train = disc_cost

            # step 2: train critic and cost several steps with respect to returns and c-returns
            critic_loss = self._update_nn_regressor(self.critic, self.critic_optim, state_train,
                                                    disc_reward_train, self.val_iter)
            cost_loss = self._update_nn_regressor(self.cost, self.cost_optim, state_train,
                                                  disc_cost_train, self.cost_iter)

            constraint_cost = (disc_cost[start_indices]).mean()

            debugs = self._update_policy(state, action, log_prob, reward_adv, cost_adv, constraint_cost)

        elapsed_time = time.time() - start_time
        debugs.update({'elapsed_time': elapsed_time})

        return {'critic_loss': critic_loss,
                'cost_loss': cost_loss}, debugs


    def _update_policy(self, state, action, log_prob_old, reward_adv, cost_adv, J_c, eps=1e-8):
        debugs = {}
        # step 3: get gradient of surrogate loss
        log_prob, mu, log_std = self.actor.log_prob(state, action, True)
        imp_sampling = torch.exp(log_prob - log_prob_old)
        reward_loss = - torch.mean(imp_sampling * reward_adv)
        reward_grad = flat_grad(reward_loss, self.actor.parameters(), retain_graph=True)
        cost_loss = torch.mean(imp_sampling * cost_adv, dim=0) # (m,)
        cost_grad = flat_grad(cost_loss, self.actor.parameters(), retain_graph=True) # (n, m)
        cost_norm = cost_grad.norm()
        debugs.update({'r_grad_norm': reward_grad.norm(),
                       'c_grad_norm': cost_norm})
        debugs.update({})

        # step 5: get step direction through conjugate gradient with fisher vector product function
        # and get step size with dual variables and get full step = step_size * step_dir
        F_inv_g = self._conjugate_gradients(state, reward_grad.data, self.cg_max_iter) # (n,)
        q = torch.matmul(reward_grad, F_inv_g)
        delta = 2 * self.max_kl

        if cost_norm < eps:
            # if safety gradient is zero, linear constraint is not present;
            # ignore linear constraint.
            optim_case = 4
            if self.verbose:
                print(f'q : {q:<5.4f}')
        else:
            F_inv_b = self._conjugate_gradients(state, cost_grad.data, self.cg_max_iter) # (n, m)
            r = torch.matmul(reward_grad, F_inv_b) # (m,)
            s = torch.matmul(cost_grad, F_inv_b) # (m, m)
            c = J_c - self.max_constraint # (m)

            A = q - r**2 / s                # this should always be positive by Cauchy-Schwarz
            B = delta - c**2 / s            # this one says whether or not the closest point on the plane is feasible

            # if (B < 0), that means the trust region plane doesn't intersect the safety boundary
            if c < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible, because it lies entirely within the linear constraint-satisfying halfspace
                # ignore linear constraint.
                optim_case = 3
            elif c < 0 and B > 0:
                # current policy is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif c > 0 and B > 0:
                # current policy is infeasible (bad! unsafe!) and safety boundary intersects
                # ==> part of trust region is feasible
                # find linear-constraint satisfying policy within trust region throuth the backtracking linesearch
                optim_case = 1
                if self.verbose:
                    print('optimizer is attempting feasible recovery')
            else:
                # current policy is infeasible (bad! unsafe!) and safety boundary doesn't intersect
                # ==> whole trust region infeasible
                optim_case = 0
                if self.verbose:
                    print('optimizer is attempting infeasible recovery')
            if self.verbose:
                print(f'q : {q:<5.4f}, r : {r:<5.4f}, s: {s:<5.4f}, c : {c:<5.4f}')


        # figure out lambda coeff (lagrange multiplier for trust region) and nu coeff (lagrange multiplier for linear constraint)
        if optim_case == 1 or optim_case == 2:
            lam, nu = self.calc_dual_vars(q, r, s, c, A, B, delta, eps)
        else:
            # default dual vars, which assume safety constraint inactive
            lam = torch.sqrt(q / delta)
            nu = 0.0
        debugs.update({'lam': lam.item(), 'nu': nu, 'optim_case': optim_case})

        if optim_case > 0:
            search_dir = - (1. / (lam + eps)) * (F_inv_g + nu * F_inv_b)
        else:
            search_dir = - torch.sqrt(delta / (s + eps)) * F_inv_b

        debug = self._linesearch(reward_loss, mu, log_std, optim_case,
                                 reward_grad, cost_grad, search_dir, c,
                                 state, action, reward_adv, log_prob_old)
        debugs.update(debug)

        return debugs


    def _linesearch(self, sur_loss_old, mu_old, log_std_old, optim_case,
                    reward_grad, cost_grad, fullstep, c,
                    states, actions, reward_adv, fixed_log_prob):
        prev_params = get_flat_params(self.actor)

        flag = False
        expected_improve = (reward_grad * fullstep).sum()
        expected_cost_improve = (cost_grad * fullstep).sum()

        for (_n_backtracks, stepfrac) in enumerate(self.line_search_coef ** np.arange(self.line_search_max_iter)):
            new_params = prev_params + stepfrac * fullstep
            set_params(self.actor, new_params)

            with torch.no_grad():
                log_prob, mu, log_std = self.actor.log_prob(states, actions, True)
                imp_sampling = torch.exp(log_prob - fixed_log_prob)
                sur_loss = - torch.mean(imp_sampling * reward_adv)

                actual_improve = sur_loss - sur_loss_old
                expected_improve *= stepfrac
                ratio = actual_improve / expected_improve
                reward_cond = ratio >= self.accept_ratio and actual_improve <= 0

                kl = self._kl_divergence(mu, log_std, mu_old, log_std_old).mean()
                kl_cond = kl <= self.max_kl

                expected_cost_improve *= stepfrac
                cost_cond = expected_cost_improve <= max(-c, 0.0)

            if self.verbose:
                print(f'step : {_n_backtracks} - (act, exp, ratio) : ( {actual_improve.item():<5.4f}, {expected_improve.item():<5.4f}' \
                      f', {ratio.item():<5.4f} ) , r_cond : {reward_cond.detach().cpu().numpy()}' \
                      f', kld_cond : {kl_cond.detach().cpu().numpy()}, c_cond : {cost_cond.detach().cpu().numpy()}')

            flag = kl_cond and cost_cond if optim_case == 0 else reward_cond and kl_cond and cost_cond
            if flag:
                break

        if not flag:
            set_params(self.actor, prev_params)

        debug = {"policy_update" : flag,
                 "backtrack_step" : _n_backtracks,
                 "sur_loss_before" : sur_loss_old.item(),
                 "sur_loss_after" : sur_loss.item(),
                 "act_improve": actual_improve.item(),
                 "exp_improve": expected_improve.item(),
                 "exp_c_improve" : expected_cost_improve.sum().item(),
                 "imp_ratio" : ratio.item(),
                 "kld" : kl.item()}

        return debug


    def calc_dual_vars(self, q, r, s, c, A, B, delta, eps):
        """
        dual function is piecewise continuous
        on region (a):

          L(lam) = -1/2 (A / lam + B * lam) - r * c / s

        on region (b):

          L(lam) = -1/2 (q / lam + delta * lam)
        """
        lam_mid = r / c
        L_mid = - 0.5 * (q / lam_mid + lam_mid * delta)

        lam_a = torch.sqrt(A / (B + eps))
        L_a = -torch.sqrt(A*B) - r*c / (s + eps)
        # note that for optim_case == 1 or 2, B > 0, so this calculation should never be an issue

        lam_b = torch.sqrt(q / delta)
        L_b = -torch.sqrt(q * delta)

        #those lam's are solns to the pieces of piecewise continuous dual function.
        #the domains of the pieces depend on whether or not c < 0 (x=0 feasible),
        #and so projection back on to those domains is determined appropriately.
        if lam_mid > 0:
            if c < 0:
                # here, domain of (a) is [0, lam_mid)
                # and domain of (b) is (lam_mid, infty)
                if lam_a > lam_mid:
                    lam_a = lam_mid
                    L_a   = L_mid
                if lam_b < lam_mid:
                    lam_b = lam_mid
                    L_b   = L_mid
            else:
                # here, domain of (a) is (lam_mid, infty)
                # and domain of (b) is [0, lam_mid)
                if lam_a < lam_mid:
                    lam_a = lam_mid
                    L_a   = L_mid
                if lam_b > lam_mid:
                    lam_b = lam_mid
                    L_b   = L_mid

            if L_a >= L_b:
                lam = lam_a
            else:
                lam = lam_b

        else:
            if c < 0:
                lam = lam_b
            else:
                lam = lam_a

        nu = max(0, lam * c - r) / (s + eps)

        if self.verbose:
            print(f'lam_mid: {lam_mid:<5.4f}, lam_a: {lam_a:<5.4f}, lam_b: {lam_b:<5.4f}')
            print(f'f_mid: {L_mid:<5.4f}, f_a: {L_a:<5.4f}, f_b: {L_b:<5.4f}')
            print(f's_inv: {(1 / s):<5.4f}')
        return lam, nu


    def get_optimizer(self):
        optim = super(CPO, self).get_optimizer()
        optim.update({"cost": self.cost_optim})
        return optim