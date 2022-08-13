import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from algos.base import Agent
from layers.graph_social_layer import GraphSocialEncoder
from layers.prediction_layer import Predictor
from layers.seq2seq import Seq2Seq
from utils.misc import prLightPurple


class GRIP(Agent):
    def __init__(self, state_space, action_space, args):
        super(GRIP, self).__init__(state_space, action_space, args)

        self._set_args(args)

        self._build_network(args)

        if args.mode == 'train' or 'pretrain':
            self._build_optimizer(args)

    def select_action(self, state):
        node_feature = torch.from_numpy(np.expand_dims(state['graph_feature'], 0)) # (N, C, T, V)
        adj = torch.from_numpy(np.expand_dims(state['adjacency'], 0)) # (N, max_hop + 1, V, V)

        node_feature = node_feature.to(self.device).type(torch.float32)[:,[0,1,-1]]
        adj = adj.to(self.device).type(torch.float32)

        node_feature, ori_node_feature = self.preprocess(node_feature)

        predicted_mask = node_feature[:,-1,-1, :].detach().cpu().numpy()
        last_loc = self.gse.reshape_for_lstm(node_feature[:,:2])[:,-1:,:] # (N*V, 1, 2)
        enc_hidden = self.gse(node_feature, adj)
        predicted = self.tp(last_loc, enc_hidden)
        ori_output_last_loc = ori_node_feature[:, :2, self.tp.pred_length-1:self.tp.pred_length, :]
        predicted = predicted * self.rescale_xy
        for ind in range(1, predicted.shape[-2]):
            predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2) # smoothing with window size 2
        predicted += ori_output_last_loc # displacement vector to location
        predicted = predicted.detach().cpu().numpy()

        return predicted, predicted_mask

    def update(self):
        pass

    def reset(self):
        pass

    def get_optimizer(self):
        optimizer = {'tp': self.tp_optim}
        return optimizer

    def train_epoch(self, loader):
        loss = 0
        for iteration, (node_feature, adjacency, gt_node_feature) in enumerate(loader):
            input_data = node_feature.to(self.device)[:,[0,1,-1]] # (N, C, T, V)
            adjacency = adjacency.to(self.device) #(N, L, V, V)
            loc_GT = gt_node_feature.to(self.device)

            input_data, _ = self.preprocess(node_feature) # (N, 2, T, V)
            loc_GT, _ = self.preprocess(gt_node_feature)
            loc_GT = loc_GT[:,:2,...]

            output_mask = input_data[:,-1:,:,:] # (N, 1, T, V)
            last_loc = self.gse.reshape_for_lstm(input_data[:,:2])[:,-1:,:] # (N*V, 1, 2)

            enc_hidden = self.gse(input_data, adjacency)
            predicted = self.tp(last_loc, enc_hidden, self.gse.reshape_for_lstm(loc_GT))

            tp_loss = self.prediction_error(predicted, loc_GT, output_mask, error_order = self.error_order)
            loss += tp_loss.item()
            self.total_it += iteration + 1

            self.tp_optim.zero_grad()
            tp_loss.backward()
            nn.utils.clip_grad_norm_(
                list(itertools.chain(*[g['params'] for g in self.tp_optim.param_groups])),
                self.max_grad_norm)
            self.tp_optim.step()

            # prLightPurple(f'\tIter : {iteration:<5} | \tLoss:{tp_loss.item():<5.2f}|')

        return {'train_tp_loss' : loss / (iteration + 1)}, {}

    def validate_epoch(self, loader):
        debug = {'ADE': [], 'FDE': []}
        for iteration, (node_feature, adjacency, gt_node_feature) in enumerate(loader):

            input_data = node_feature.to(self.device)[:,[0,1,-1]] # (N, C, T, V)
            adjacency = adjacency.to(self.device) #(N, L, V, V)
            gt_node_feature = gt_node_feature[:,:2,...].to(self.device)

            input_data, ori_node_feature = self.preprocess(node_feature) # (N, 2, T, V)

            output_mask = input_data[:,-1:,:,:] # (N, 1, T, V)
            last_loc = self.gse.reshape_for_lstm(input_data[:,:2])[:,-1:,:] # (N*V, 1, 2)
            enc_hidden = self.gse(input_data, adjacency)

            predicted = self.tp(last_loc, enc_hidden)
            ori_output_last_loc = ori_node_feature[:, :2, self.tp.pred_length-1:self.tp.pred_length, :]
            predicted = predicted * self.rescale_xy
            for ind in range(1, predicted.shape[-2]):
                predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2) # smoothing with window size 2
            predicted += ori_output_last_loc # displacement vector to location

            criterion = self.criterion(predicted, gt_node_feature, output_mask)
            for k, v in criterion.items():
                debug[k].extend([v])

        return {'ADE_mean': np.mean(debug['ADE']), 'ADE_std': np.std(debug['ADE']),
                'FDE_mean': np.mean(debug['FDE']), 'FDE_std': np.std(debug['FDE'])}

    def preprocess(self, pra_data):
        # pra_data: (N, C, T, V)
        # C = 10: [position_x, position_y, heading, vx, vy, ax, ay, wx, wy, mask]
        ori_data = pra_data
        data = ori_data.clone()

        new_mask = (data[:, :2, 1:]!=0) * (data[:, :2, :-1]!=0)
        data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float() # use displacement vector
        data[:, :2, 0] = 0

        data[:,:2] = data[:,:2] / self.rescale_xy # normalize displacement vector

        return data, ori_data

    def prediction_error(self, pred, GT, output_mask, error_order=1):
        pred = pred * output_mask # (N, 2, T, V)
        GT = GT * output_mask # (N, 2, T, V)
        error_order = 1
        error = torch.sum(torch.abs(pred - GT)**error_order, dim=1) # (N, C, T, V) -> (N, T, V)

        overall_sum_time = error.sum(dim=-1) # (N, T, V) -> (N, T)
        overall_mask = output_mask.sum(dim=1).sum(dim=-1) # (N, C, T, V) -> (N, T)
        pred_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_mask),
                                                            torch.ones(1,).to(self.device)) #(1,)
        return pred_loss

    def criterion(self, pred, GT, output_mask):
        pred = pred * output_mask # (N, 2, T, V)
        GT = GT * output_mask # (N, 2, T, V)
        distance_error = torch.linalg.norm(pred - GT, dim=1) # (N, T, V)
        final_distance_error = distance_error[:, -1] # # (N, V)

        return {'ADE': torch.mean(distance_error).item(),
                'FDE': torch.mean(final_distance_error).item()}

    def _set_args(self, args):
        self.rescale_xy = torch.ones((1,2,1,1)).to(args.device)
        self.rescale_xy[:,0] = args.max_x
        self.rescale_xy[:,1] = args.max_y
        self.error_order = args.error_order

    def _build_network(self, args):        
        self.gse = GraphSocialEncoder(in_channels=self.state_dim,
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

    def _build_optimizer(self, args):
        self.tp_optim = Adam(list(self.gse.parameters()) + list(self.tp.parameters()), lr=args.tp_lr)

class VanilaGRU(GRIP):
    def select_action(self, state):
        node_feature = torch.from_numpy(np.expand_dims(state['graph_feature'], 0)) # (N, C, T, V)
        node_feature = node_feature.to(self.device).type(torch.float32)[:,[0,1,-1]]

        predicted_mask = node_feature[:,-1,-1, :].detach().cpu().numpy()
        last_loc = self.seq2seq.reshape_for_lstm(node_feature[:,:2])[:,-1:,:] # (N*V, 1, 2)
        node_feature = self.seq2seq.reshape_for_lstm(node_feature) # (N*V, T, C)

        predicted = self.seq2seq(node_feature, last_loc)
        predicted = self.seq2seq.reshape_from_lstm(predicted) # (N, 2, T, V)
        predicted = predicted.detach().cpu().numpy()

        return predicted, predicted_mask

    def train_epoch(self, loader):
        loss = 0
        for iteration, (node_feature, _, gt_node_feature) in enumerate(loader):
            input_data = node_feature.to(self.device)[:,[0,1,-1]] # (N, C, T, V)
            loc_GT = gt_node_feature.to(self.device)[:,:2,...]

            output_mask = input_data[:,-1:,:,:] # (N, 1, T, V)
            last_loc = self.seq2seq.reshape_for_lstm(input_data[:,:2])[:,-1:,:] # (N*V, 1, 2)
            input_data = self.seq2seq.reshape_for_lstm(input_data) # (N*V, T, C)

            predicted = self.seq2seq(input_data, last_loc, self.seq2seq.reshape_for_lstm(loc_GT))
            predicted = self.seq2seq.reshape_from_lstm(predicted) # (N, 2, T, V)

            tp_loss = self.prediction_error(predicted, loc_GT, output_mask, error_order = self.error_order)
            loss += tp_loss.item()
            self.total_it += iteration + 1

            self.tp_optim.zero_grad()
            tp_loss.backward()
            nn.utils.clip_grad_norm_(
                list(itertools.chain(*[g['params'] for g in self.tp_optim.param_groups])),
                self.max_grad_norm)
            self.tp_optim.step()

            # prLightPurple(f'\tIter : {iteration:<5} | \tLoss:{tp_loss.item():<5.2f}|')

        return {'train_tp_loss' : loss / (iteration + 1)}, {}

    def validate_epoch(self, loader):
        debug = {'ADE': [], 'FDE': []}
        for iteration, (node_feature, _, gt_node_feature) in enumerate(loader):

            input_data = node_feature.to(self.device)[:,[0,1,-1]] # (N, C, T, V)
            gt_node_feature = gt_node_feature[:,:2,...].to(self.device)

            output_mask = input_data[:,-1:,:,:] # (N, 1, T, V)
            last_loc = self.seq2seq.reshape_for_lstm(input_data[:,:2])[:,-1:,:] # (N*V, 1, 2)
            input_data = self.seq2seq.reshape_for_lstm(input_data) # (N*V, T, C)

            predicted = self.seq2seq(input_data, last_loc)
            predicted = self.seq2seq.reshape_from_lstm(predicted) # (N, 2, T, V)

            criterion = self.criterion(predicted, gt_node_feature, output_mask)
            for k, v in criterion.items():
                debug[k].extend([v])

        return {'ADE_mean': np.mean(debug['ADE']), 'ADE_std': np.std(debug['ADE']),
                'FDE_mean': np.mean(debug['FDE']), 'FDE_std': np.std(debug['FDE'])}


    def _build_network(self, args):
        self.state_dim = 3
        self.seq2seq = Seq2Seq(input_size=self.state_dim,
                               hidden_size=args.rnn_hidden_size,
                               num_layers=args.rnn_num_layer,
                               pred_length=args.num_frame,
                               num_node=args.max_object,
                               output_size=2,
                               dropout=args.dropout,
                               teacher_forcing_ratio=args.teacher_forcing_ratio)

    def _build_optimizer(self, args):
        self.tp_optim = Adam(self.seq2seq.parameters(), lr=args.tp_lr)
