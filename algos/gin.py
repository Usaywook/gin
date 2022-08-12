import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from gym.spaces import Dict
from algos.wcsac import SAC, WCSAC
from layers.concatenator import Concatenator
from utils import soft_update, hard_update, prRed

from layers.graph_social_layer import GraphSocialEncoder
from layers.prediction_layer import Predictor
from layers.seq2seq import EncoderRNN
from models.hmm import HMMEncoder

class GIN(WCSAC):
    def __init__(self, state_space, action_space, args):
        self.dict_obs = isinstance(state_space, Dict)
        super(SAC, self).__init__(state_space['feature'] if self.dict_obs else state_space,
                                  action_space, args)
        self._set_args(args)

        self._build_network(state_space, action_space, args)

        if args.mode == 'train':
            self._build_optimizer(args)

    def select_action(self, state):
        kinetic_feature = torch.from_numpy(state['feature']).to(self.device).type(torch.float32) # (N, H1)
        node_feature = torch.from_numpy(state['graph_feature']).to(self.device).type(torch.float32) # (N, C, T, V)
        adj = torch.from_numpy(state['adjacency']).to(self.device).type(torch.float32) # (N, L + 1, V, V)
        node_feature, ori_node_feature = self._preprocess(node_feature)
        ori_last_loc = ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()

        with torch.no_grad():
            if self.hazard:
                # Visualize predicted trajectory
                self.predicted_mask = node_feature[:,-1,-1, :].detach().cpu().numpy()
                last_loc = self.gse.reshape_for_lstm(node_feature[:,:2])[:,-1:,:] # (N*V, 1, 2)
                dynamic_feature, enc_hidden = self.gse(node_feature, adj, ret_h=True, ret_o=True) # (N, 2, V) -> (N, V, 2)
                predicted = self.tp(last_loc, enc_hidden)
                ori_output_last_loc = ori_node_feature[:, :2, self.tp.pred_length-1:self.tp.pred_length, :] #(N, 2, 1, V)
                predicted = predicted * self.rescale_xy
                for ind in range(1, predicted.shape[-2]):
                    predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2) # smoothing with window size 2
                predicted += ori_output_last_loc # displacement vector to location
                self.predicted = predicted.detach().cpu().numpy() #(N, 2, T, V)
            else:
                dynamic_feature = self.gse(node_feature, adj, ret_o=True)

            state = self.base(kinetic_feature, dynamic_feature, ori_last_loc=ori_last_loc) # (N, H)

        if self.is_training:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)

        action = action.detach().cpu().data.numpy().flatten()

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
            tp_mini_batch = storage['tp_buffer']._encode_sample(idxes)
        else:
            mini_batch = storage['buffer'].sample(self.batch_size)
            tp_mini_batch = storage['tp_buffer'].sample(self.batch_size)

        tp_node_feature, tp_adjacency = self._sample_from_tp_batch(tp_mini_batch)
        tp_loss = self._update_social_regressor(tp_node_feature, tp_adjacency)

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
                  "alpha": self.alpha,
                  "kappa_loss": kappa_loss,
                  "kappa": self.kappa,
                  "tp_loss": tp_loss}

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
        super(GIN, self)._set_args(args)

        self.rescale_xy = torch.ones((1,2,1,1)).to(args.device)
        self.rescale_xy[:,0] = args.max_x
        self.rescale_xy[:,1] = args.max_y

        self.hazard = args.hazard
        self.error_order = args.error_order

    def _build_network(self, state_space, action_space, args):
        super(GIN, self)._build_network(state_space, action_space, args)
        graph_in_channels = state_space['graph_feature'].shape[0]

        self.gse = GraphSocialEncoder(in_channels=graph_in_channels,
                                      max_hop=args.max_hop,
                                      num_node=args.max_object,
                                      edge_importance_weighting=args.trainable_edge,
                                      temporal_kernel_size=args.temporal_kernel_size,
                                      graph_hidden_size=args.graph_hidden_size,
                                      rnn_hidden_size=args.rnn_hidden_size,
                                      rnn_num_layer=args.rnn_num_layer)

        self.tp = Predictor(pred_length=args.num_frame,
                            num_node=args.max_object,
                            teacher_forcing_ratio=args.teacher_forcing_ratio,
                            hidden_size=args.rnn_hidden_size,
                            output_size=2,
                            num_layers=args.rnn_num_layer,
                            dropout=args.dropout)

        self.base = Concatenator(input_dim=(self.state_dim, args.rnn_hidden_size + 2),
                                 hidden_dim=args.hidden_size,
                                 distpool=args.distpool,
                                 sigma=args.sigma)

    def _build_optimizer(self, args):
        parameters = list(self.critic.parameters())
        parameters += list(self.cost.parameters())
        parameters += list(self.cost_var.parameters())
        # parameters += list(self.gse.parameters())
        parameters += list(self.base.parameters())
        self.critic_optim = Adam(parameters, lr=args.lr)

        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.kappa_optim = Adam([self.log_kappa], lr=args.lr)

        self.tp_optim = Adam(list(self.gse.parameters()) + list(self.tp.parameters()), lr=args.tp_lr)

    def _sample_from_tp_batch(self, tp_mini_batch):
        tp_node_feature = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['tp_graph_feature'], tp_mini_batch)))).type(torch.float32).to(self.device)
        tp_adjacency = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['tp_adjacency'], tp_mini_batch)))).type(torch.float32).to(self.device)
        return tp_node_feature, tp_adjacency

    def _sample_from_batch(self, mini_batch):
        kinetic_feature = torch.from_numpy(np.vstack(list(map(lambda x: x.item()['feature'], mini_batch[0])))).type(torch.float32).to(self.device)
        node_feature = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['graph_feature'], mini_batch[0])))).type(torch.float32).to(self.device)
        adjacency = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['adjacency'], mini_batch[0])))).type(torch.float32).to(self.device)
        action = torch.from_numpy(np.vstack(mini_batch[1])).type(torch.float32).to(self.device)
        reward = torch.from_numpy(np.vstack(mini_batch[2])).type(torch.float32).to(self.device)
        next_kinetic_feature = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['feature'], mini_batch[3])))).type(torch.float32).to(self.device)
        next_node_feature = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['graph_feature'], mini_batch[3])))).type(torch.float32).to(self.device)
        next_adjacency = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['adjacency'], mini_batch[3])))).type(torch.float32).to(self.device)
        mask = torch.from_numpy(1 - np.vstack(mini_batch[4])).type(torch.float32).to(self.device)
        cost = torch.from_numpy(np.vstack(mini_batch[5])).type(torch.float32).to(self.device)

        node_feature, ori_node_feature = self._preprocess(node_feature)
        ori_last_loc = ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()
        next_node_feature, next_ori_node_feature = self._preprocess(next_node_feature)
        next_ori_last_loc = next_ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()

        with torch.no_grad():
            dynamic_feature = self.gse(node_feature, adjacency, ret_o=True)
            next_dynamic_feature = self.gse(next_node_feature, next_adjacency, ret_o=True)
        state = self.base(kinetic_feature, dynamic_feature, ori_last_loc=ori_last_loc)        
        next_state = self.base(next_kinetic_feature, next_dynamic_feature, ori_last_loc=next_ori_last_loc)

        return state, action, reward, mask, next_state, cost

    def _update_social_regressor(self, node_feature, adjacency):
        tp_graph_feature, _ = self._preprocess(node_feature) # (N, C, T, V)
        loc_GT = tp_graph_feature[:,:2,self.tp.pred_length:,:] # (N, 2, T, V)
        output_mask = tp_graph_feature[:,-1:,self.tp.pred_length:,:] # (N, 1, T, V)
        last_loc = self.gse.reshape_for_lstm(tp_graph_feature[:,:2,:self.tp.pred_length])[:,-1:,:] # (N*V, 1, 2)

        enc_hidden = self.gse(tp_graph_feature[:,:,:self.tp.pred_length], adjacency)
        predicted = self.tp(last_loc, enc_hidden, self.gse.reshape_for_lstm(loc_GT))
        tp_loss = self._prediction_error(predicted, loc_GT, output_mask, error_order = self.error_order)

        self.tp_optim.zero_grad()
        tp_loss.backward()
        nn.utils.clip_grad_norm_(
            list(itertools.chain(*[g['params'] for g in self.tp_optim.param_groups])),
            self.max_grad_norm)
        self.tp_optim.step()

        return tp_loss.item()

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
        optimizer = super(GIN, self).get_optimizer()
        optimizer.update({'tp': self.tp_optim})

        return optimizer

    def validate_epoch(self, loader):
        debug = {'ADE': [], 'FDE': []}
        for iteration, (node_feature, adjacency, gt_node_feature) in enumerate(loader):
            node_feature = node_feature.to(self.device) # (N, C, T, V)
            adjacency = adjacency.to(self.device) #(N, L, V, V)
            gt_node_feature = gt_node_feature[:,:2,...].to(self.device)

            input_data, ori_node_feature = self._preprocess(node_feature) # (N, 2, T, V)

            output_mask = input_data[:,-1:,:,:] # (N, 1, T, V)
            last_loc = self.gse.reshape_for_lstm(input_data[:,:2])[:,-1:,:] # (N*V, 1, 2)
            enc_hidden = self.gse(input_data, adjacency)

            predicted = self.tp(last_loc, enc_hidden)
            ori_output_last_loc = ori_node_feature[:, :2, self.tp.pred_length-1:self.tp.pred_length, :]
            predicted = predicted * self.rescale_xy
            for ind in range(1, predicted.shape[-2]):
                predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2) # smoothing with window size 2
            predicted += ori_output_last_loc # displacement vector to location

            criterion = self._criterion(predicted, gt_node_feature, output_mask)
            for k, v in criterion.items():
                debug[k].extend([v])

        return {'ADE_mean': np.mean(debug['ADE']), 'ADE_std': np.std(debug['ADE']),
                'FDE_mean': np.mean(debug['FDE']), 'FDE_std': np.std(debug['FDE'])}

    def _criterion(self, pred, GT, output_mask):
        pred = pred * output_mask # (N, 2, T, V)
        GT = GT * output_mask # (N, 2, T, V)
        distance_error = torch.linalg.norm(pred - GT, dim=1) # (N, T, V)
        final_distance_error = distance_error[:, -1] # # (N, V)

        return {'ADE': torch.mean(distance_error).item(),
                'FDE': torch.mean(final_distance_error).item()}

class Ably(GIN):
    def select_action(self, state):
        kinetic_feature = torch.from_numpy(state['feature']).to(self.device).type(torch.float32) # (N, H1)
        node_feature = torch.from_numpy(state['graph_feature']).to(self.device).type(torch.float32) # (N, C, T, V)
        adj = torch.from_numpy(state['adjacency']).to(self.device).type(torch.float32) # (N, L + 1, V, V)
        node_feature, ori_node_feature = self._preprocess(node_feature)
        ori_last_loc = ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()

        with torch.no_grad():
            if self.study == 'temporal':
                _, dynamic_feature = self.encoder(self._reshape_for_lstm(node_feature)) # (L, N*V, H)
                dynamic_feature = self._reshape_for_context(dynamic_feature[-1]) # (N, V, H)
            elif self.study == 'social':
                dynamic_feature = self.encoder(node_feature, adj, ret_o=True)
            elif self.study == 'hmm':
                dynamic_feature = self.encoder(node_feature)
            elif self.study == 'random':
                dynamic_feature = torch.randn(node_feature.shape[0], self.num_node, self.num_state, device=self.device)

            state = self.base(kinetic_feature, dynamic_feature, ori_last_loc=ori_last_loc) # (N, H)

        if self.is_training:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)

        action = action.detach().cpu().data.numpy().flatten()

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
                  "alpha": self.alpha,
                  "kappa_loss": kappa_loss,
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

    def fit_encoder(self, data_path):
        batch_indices = np.random.choice(range(self.num_sample), self.num_sample)
        node_feature = torch.load(data_path)['graph_feature'].to(self.device)[batch_indices,:,:self.num_frame,:]
        node_feature, _ = self._preprocess(node_feature)
        self.encoder.update(node_feature)

    def _sample_from_batch(self, mini_batch):
        kinetic_feature = torch.from_numpy(np.vstack(list(map(lambda x: x.item()['feature'], mini_batch[0])))).type(torch.float32).to(self.device)
        node_feature = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['graph_feature'], mini_batch[0])))).type(torch.float32).to(self.device)
        adjacency = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['adjacency'], mini_batch[0])))).type(torch.float32).to(self.device)
        action = torch.from_numpy(np.vstack(mini_batch[1])).type(torch.float32).to(self.device)
        reward = torch.from_numpy(np.vstack(mini_batch[2])).type(torch.float32).to(self.device)
        next_kinetic_feature = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['feature'], mini_batch[3])))).type(torch.float32).to(self.device)
        next_node_feature = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['graph_feature'], mini_batch[3])))).type(torch.float32).to(self.device)
        next_adjacency = torch.from_numpy(np.vstack(list(map(lambda x : x.item()['adjacency'], mini_batch[3])))).type(torch.float32).to(self.device)
        mask = torch.from_numpy(1 - np.vstack(mini_batch[4])).type(torch.float32).to(self.device)
        cost = torch.from_numpy(np.vstack(mini_batch[5])).type(torch.float32).to(self.device)

        node_feature, ori_node_feature = self._preprocess(node_feature)
        ori_last_loc = ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()
        next_node_feature, next_ori_node_feature = self._preprocess(next_node_feature)
        next_ori_last_loc = next_ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()

        if self.study == 'temporal':
            _, dynamic_feature = self.encoder(self._reshape_for_lstm(node_feature)) # (L, N*V, H)
            dynamic_feature = self._reshape_for_context(dynamic_feature[-1]) # (N, V, H)
            _, next_dynamic_feature = self.encoder(self._reshape_for_lstm(next_node_feature)) # (L, N*V, H)
            next_dynamic_feature = self._reshape_for_context(next_dynamic_feature[-1]) # (N, V, H)
        elif self.study == 'social':
            dynamic_feature = self.encoder(node_feature, adjacency, ret_o=True)
            next_dynamic_feature = self.encoder(next_node_feature, next_adjacency, ret_o=True)
        elif self.study == 'hmm':
            dynamic_feature = self.encoder(node_feature)
            next_dynamic_feature = self.encoder(next_node_feature)
        elif self.study == 'random':
            dynamic_feature = torch.randn(node_feature.shape[0], self.num_node, self.num_state, device=self.device)
            next_dynamic_feature = torch.randn(next_node_feature.shape[0], self.num_node, self.num_state, device=self.device)

        state = self.base(kinetic_feature, dynamic_feature, ori_last_loc=ori_last_loc)
        with torch.no_grad():
            next_state = self.base(next_kinetic_feature, next_dynamic_feature, ori_last_loc=next_ori_last_loc)

        return state, action, reward, mask, next_state, cost

    def _set_args(self, args):
        super(Ably, self)._set_args(args)
        self.study = args.study
        self.total_epoch = 0
        self.num_node = args.max_object
        self.num_frame = args.num_frame
        self.num_state = args.num_state
        self.num_sample = args.num_sample
        if self.study == 'hmm' or 'random':
            args.rnn_hidden_size = self.num_state

    def _build_network(self, state_space, action_space, args):
        super(GIN, self)._build_network(state_space, action_space, args)
        graph_in_channels = state_space['graph_feature'].shape[0]

        if self.study == 'temporal':
            self.encoder = EncoderRNN(input_size=graph_in_channels,
                                      hidden_size=args.rnn_hidden_size,
                                      num_layers=args.rnn_num_layer)
        elif self.study == 'social':
            self.encoder = GraphSocialEncoder(in_channels=graph_in_channels,
                                              max_hop=args.max_hop,
                                              num_node=args.max_object,
                                              edge_importance_weighting=args.trainable_edge,
                                              temporal_kernel_size=args.temporal_kernel_size,
                                              graph_hidden_size=args.graph_hidden_size,
                                              rnn_hidden_size=args.rnn_hidden_size,
                                              rnn_num_layer=args.rnn_num_layer)
        elif self.study == 'hmm':
            self.encoder = HMMEncoder(num_state=args.num_state, n_iter=args.n_iter)

        self.base = Concatenator(input_dim=(self.state_dim, args.rnn_hidden_size + 2),
                                 hidden_dim=args.hidden_size,
                                 distpool=args.distpool,
                                 sigma=args.sigma)

    def _build_optimizer(self, args):
        parameters = list(self.critic.parameters())
        parameters += list(self.cost.parameters())
        parameters += list(self.cost_var.parameters())
        if self.study == 'temporal' or self.study == 'social':
            parameters += list(self.encoder.parameters())
        parameters += list(self.base.parameters())
        self.critic_optim = Adam(parameters, lr=args.lr)

        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.kappa_optim = Adam([self.log_kappa], lr=args.lr)

    def get_optimizer(self):
        optimizer = super(GIN, self).get_optimizer()
        return optimizer

    def _reshape_for_lstm(self, feature):
		# prepare for skeleton prediction model
        '''
        N: batch_size
        C: channel
        T: time_step
        V: nodes
        '''
        N, C, T, V = feature.size()
        now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
        now_feat = now_feat.view(N*V, T, C)
        return now_feat

    def _reshape_for_context(self, feature):
        NV, H = feature.size()
        now_feat = feature.view(-1, self.num_node, H) # (N, V, H)
        return now_feat
