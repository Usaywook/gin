import time
import itertools

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam

from algos.cpo import CPO
from utils.misc import prRed, get_flat_params
from models.actor import GaussianPolicy
from models.critic import StateValueFunction
from layers.prediction_layer import Predictor
from layers.graph_social_layer import GraphSocialEncoder
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class CGRAIN(CPO):
    def __init__(self, state_space, action_space, args):
        args.hidden_size = args.rnn_output_size
        super(CGRAIN, self).__init__(state_space, action_space, args)
        # input_size = self.state_dim + args.rnn_output_size if self.dict_obs else self.state_dim
        # self.actor = GaussianPolicy(input_size, self.action_dim, args.hidden_size, action_space)
        # hidden_units = (args.hidden_size, args.hidden_size)
        # self.critic = StateValueFunction(input_size, hidden_units=hidden_units)
        # self.cost = StateValueFunction(input_size, hidden_units=hidden_units)
        graph_in_channels = state_space['graph_feature'].shape[0]

        self.gse = GraphSocialEncoder(in_channels=graph_in_channels,
                                      max_hop=args.max_hop,
                                      num_node=args.max_object,
                                      edge_importance_weighting=args.trainable_edge,
                                      temporal_kernel_size=args.temporal_kernel_size,
                                      graph_hidden_size=args.graph_hidden_size,
                                      rnn_hidden_size=args.rnn_hidden_size,
                                      rnn_num_layer=args.rnn_num_layer,
                                      rnn_output_size=args.rnn_output_size)

        self.tp = Predictor(pred_length=args.num_frame,
                            num_node=args.max_object,
                            teacher_forcing_ratio=args.teacher_forcing_ratio,
                            hidden_size=args.rnn_hidden_size,
                            output_size=2,
                            num_layers=args.rnn_num_layer,
                            dropout=args.dropout)

        # self.critic_optim.add_param_group({'params':self.gse.linear.parameters(), 'lr': args.linear_lr})
        self.cost_optim.add_param_group({'params':self.gse.linear.parameters(), 'lr': args.linear_lr})
        self.tp_optim = Adam(list(self.gse.parameters()) + list(self.tp.parameters()), lr=args.tp_lr)

        self.rescale_xy = torch.ones((1,2,1,1)).to(args.device)
        self.rescale_xy[:,0] = args.max_x
        self.rescale_xy[:,1] = args.max_y

        self.hazard = args.hazard
        self.tp_epoch = args.tp_epoch
        self.error_order = args.error_order


    def select_action(self, state):
        path_feature = torch.from_numpy(state['feature']).to(self.device).type(torch.float32) # (N, H1)
        node_feature = torch.from_numpy(state['graph_feature']).to(self.device).type(torch.float32) # (N, C, T, V)
        adj = torch.from_numpy(state['adjacency']).to(self.device).type(torch.float32) # (N, L + 1, V, V)
        node_feature, ori_node_feature = self._preprocess(node_feature)
        ori_last_loc = ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()

        if self.hazard:
            # Visualize predicted trajectory
            self.predicted_mask = node_feature[:,-1,-1, :].detach().cpu().numpy()
            last_loc = self.gse.reshape_for_lstm(node_feature[:,:2])[:,-1:,:] # (N*V, 1, 2)
            global_feature, enc_hidden = self.gse(node_feature, adj, ret_h=True, ori_last_loc=ori_last_loc) # (N, 2, V) -> (N, V, 2)
            predicted = self.tp(last_loc, enc_hidden)
            ori_output_last_loc = ori_node_feature[:, :2, self.tp.pred_length-1:self.tp.pred_length, :] #(N, 2, 1, V)
            predicted = predicted * self.rescale_xy
            for ind in range(1, predicted.shape[-2]):
                predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2) # smoothing with window size 2
            predicted += ori_output_last_loc # displacement vector to location
            self.predicted = predicted.detach().cpu().numpy()
        else:
            global_feature = self.gse(node_feature, adj, ori_last_loc=ori_last_loc)

        state = torch.cat([path_feature, global_feature], dim=-1) # (N, H1 + H2)
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

        if N < self.batch_size:
            prRed('rollout length have to be greater than batch size')
            raise AssertionError

        mini_batch = storage['buffer'].sample()
        adjacency = torch.from_numpy(mini_batch[0]).type(torch.float32).to(self.device)
        path_feature = torch.from_numpy(mini_batch[1]).type(torch.float32).to(self.device)
        node_feature = torch.from_numpy(mini_batch[2]).type(torch.float32).to(self.device)
        tp_node_feature = torch.from_numpy(mini_batch[3]).type(torch.float32).to(self.device)
        tp_adjacency = torch.from_numpy(mini_batch[4]).type(torch.float32).to(self.device)

        action = torch.from_numpy(mini_batch[5]).type(torch.float32).to(self.device)
        reward = torch.from_numpy(mini_batch[6]).type(torch.float32).to(self.device)
        cost = torch.from_numpy(mini_batch[7]).type(torch.float32).to(self.device)
        log_prob = torch.from_numpy(mini_batch[8]).type(torch.float32).to(self.device)
        done = mini_batch[9]
        start_indices = np.concatenate([[0], np.where(done)[0][:-1] + 1])
        done = torch.from_numpy(done).type(torch.float32).to(self.device)

        start_time = time.time()

        node_feature, ori_node_feature = self._preprocess(node_feature)
        last_loc = ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()

        for epoch in range(self.n_epoch):
            with torch.no_grad():
                global_feature = self.gse(node_feature, adjacency, ori_last_loc=last_loc)
                state = torch.cat([path_feature, global_feature], dim=-1)

                pred_value = self.critic(state)
                pred_cost = self.cost(state)

            social_loss = self._update_social_regressor(tp_node_feature, tp_adjacency)

            disc_reward, reward_adv = self._get_ret_gae(reward, 1 - done, pred_value, self.discount, self.lambd)
            disc_cost, cost_adv = self._get_ret_gae(cost, 1 - done, pred_cost, self.discount, self.lambd_c)

            if self.fusion:
                if self.state_prev is not None:
                    state_train = tuple(map(lambda x : torch.cat([*x]), zip((node_feature,
                                                                                    adjacency,
                                                                                    last_loc,
                                                                                    path_feature),
                                                                                   self.state_prev)))
                    disc_reward_train = torch.cat([disc_reward, self.disc_reward_prev])
                    disc_cost_train = torch.cat([disc_cost, self.disc_cost_prev])

                else:
                    state_train = (node_feature, adjacency, last_loc, path_feature)
                    disc_reward_train = disc_reward
                    disc_cost_train = disc_cost

                self.state_prev = (node_feature, adjacency, last_loc, path_feature)
                self.disc_reward_prev = disc_reward
                self.disc_cost_prev = disc_cost
            else:
                state_train = (node_feature, adjacency, last_loc, path_feature)
                disc_reward_train = disc_reward
                disc_cost_train = disc_cost

            critic_loss = self._update_nn_regressor(self.critic, self.critic_optim, state_train,
                                                   disc_reward_train, self.val_iter)
            cost_loss = self._update_nn_regressor(self.cost, self.cost_optim, state_train,
                                                  disc_cost_train, self.cost_iter, backbone=True)

            constraint_cost = (disc_cost[start_indices]).mean()
            debugs = self._update_policy(state, action, log_prob, reward_adv, cost_adv, constraint_cost)


        elapsed_time = time.time() - start_time
        debugs.update({'elapsed_time': elapsed_time})

        return {'critic_loss': critic_loss,
                'cost_loss': cost_loss,
                'social_loss': social_loss}, debugs

    def _update_nn_regressor(self, nn_regressor, optimizer, states, targets, n_iters=1, backbone=False):
        total_loss = []
        N = states[0].shape[0]
        for _ in range(n_iters):
            total_ind = np.arange(N)
            np.random.shuffle(total_ind)
            for j in range(N // self.batch_size):
                mini_batch_ind = total_ind[self.batch_size * j : self.batch_size * (j + 1)]

                global_feature = self.gse(states[0], states[1], ori_last_loc=states[2])
                if not backbone:
                    global_feature = global_feature.detach()
                inputs = torch.cat([states[3], global_feature], dim=-1)

                predictions = nn_regressor(inputs[mini_batch_ind])
                loss = (predictions - targets[mini_batch_ind]).pow(2).mean()

                flat_params = get_flat_params(nn_regressor)
                flat_params = torch.cat([flat_params, get_flat_params(self.gse.linear)])
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


    def _update_social_regressor(self, node_feature, adjacency):
        social_loss = []
        N = node_feature.shape[0]
        total_ind = np.arange(N)
        np.random.shuffle(total_ind)
        for epoch in range(self.tp_epoch):
            for j in range(N // self.batch_size):
                mini_batch_ind = total_ind[self.batch_size * j : self.batch_size * (j + 1)]
                tp_node_feature = node_feature[mini_batch_ind]
                tp_adjacency = adjacency[mini_batch_ind]

                tp_graph_feature, _ = self._preprocess(tp_node_feature) # (N, C, T, V)
                loc_GT = tp_graph_feature[:,:2,self.tp.pred_length:,:] # (N, 2, T, V)
                output_mask = tp_graph_feature[:,-1:,self.tp.pred_length:,:] # (N, 1, T, V)
                last_loc = self.gse.reshape_for_lstm(tp_graph_feature[:,:2,:self.tp.pred_length])[:,-1:,:] # (N*V, 1, 2)

                enc_hidden = self.gse(tp_graph_feature[:,:,:self.tp.pred_length], tp_adjacency)
                predicted = self.tp(last_loc, enc_hidden, self.gse.reshape_for_lstm(loc_GT))
                tp_loss = self._prediction_error(predicted, loc_GT, output_mask, error_order = self.error_order)
                social_loss.append(tp_loss.item())

                self.tp_optim.zero_grad()
                tp_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(
                    list(itertools.chain(*[g['params'] for g in self.tp_optim.param_groups])),
                    self.max_grad_norm)
                self.tp_optim.step()

        return np.mean(social_loss)


    def _preprocess(self, pra_data):
        # pra_data: (N, C, T, V)
        # C = 10: [position_x, position_y, heading, vx, vy, ax, ay, wx, wy, mask]
        ori_data = pra_data
        data = ori_data.clone()

        new_mask = (data[:, :2, 1:]!=0) * (data[:, :2, :-1]!=0)
        data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float() # use displacement vector
        data[:, :2, 0] = 0

        data[:,:2] = data[:,:2] / self.rescale_xy # normalize displacement vector

        return data, ori_data


    def _prediction_error(self, pred, GT, output_mask, error_order=1):
        pred = pred * output_mask # (N, 2, T, V)
        GT = GT * output_mask # (N, 2, T, V)
        error_order = 1
        error = torch.sum(torch.abs(pred - GT)**error_order, dim=1) # (N, C, T, V) -> (N, T, V)

        overall_sum_time = error.sum(dim=-1) # (N, T, V) -> (N, T)
        overall_mask = output_mask.sum(dim=1).sum(dim=-1) # (N, C, T, V) -> (N, T)
        pred_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_mask),
                                                            torch.ones(1,).to(self.device)) #(1,)
        return pred_loss


    def reset(self):
        pass


    def get_optimizer(self):
        optimizer = {'critic': self.critic_optim,
                     'cost' : self.cost_optim,
                     'tp': self.tp_optim}
        return optimizer
