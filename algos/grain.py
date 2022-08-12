import os
import itertools

import wandb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gym.spaces import Dict
import torch
import torch.nn as nn
from torch.optim import Adam
from celluloid import Camera

from algos.sac import SAC
from layers.concatenator import Concatenator
from layers.graph_social_layer import GraphSocialEncoder
from layers.prediction_layer import Predictor
from utils import soft_update, prCyan, prRed


class GRAIN(SAC):
    def __init__(self, state_space, action_space, args):
        self.dict_obs = isinstance(state_space, Dict)
        super(SAC, self).__init__(state_space['feature'] if self.dict_obs else state_space,
                                  action_space, args)
        self._set_args(args)

        self._build_network(state_space, action_space, args)

        if args.mode == 'train':
            self._build_optimizer(args)

        if args.save_video:
            self.set_camera()

    def select_action(self, state):
        kinetic_feature = torch.from_numpy(state['feature']).to(self.device).type(torch.float32) # (N, H1)
        node_feature = torch.from_numpy(state['graph_feature']).to(self.device).type(torch.float32) # (N, C, T, V)
        adj = torch.from_numpy(state['adjacency']).to(self.device).type(torch.float32) # (N, max_hop + 1, V, V)
        node_feature, ori_node_feature = self._preprocess(node_feature)
        ori_last_loc = ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()

        if self.hazard:
            # # Visualize predicted trajectory
            self.predicted_mask = node_feature[:,-1,-1, :].detach().cpu().numpy()
            last_loc = self.gse.reshape_for_lstm(node_feature[:,:2])[:,-1:,:] # (N*V, 1, 2)
            dynamic_feature, enc_hidden = self.gse(node_feature, adj, ret_h=True, ret_o=True) # (N, 2, V) -> (N, V, 2)
            predicted = self.tp(last_loc, enc_hidden)
            ori_output_last_loc = ori_node_feature[:, :2, self.tp.pred_length-1:self.tp.pred_length, :] #(N, 2, 1, V)
            predicted = predicted * self.rescale_xy
            for ind in range(1, predicted.shape[-2]):
                predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2) # smoothing with window size 2
            predicted += ori_output_last_loc # displacement vector to location
            self.predicted = predicted.detach().cpu().numpy()
        else:
            dynamic_feature = self.gse(node_feature, adj, ret_o=True)

        state = self.base(kinetic_feature, dynamic_feature, ori_last_loc=ori_last_loc) # (N, H)

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
            tp_mini_batch = storage['tp_buffer']._encode_sample(idxes)
        else:
            mini_batch = storage['buffer'].sample(self.batch_size)
            tp_mini_batch = storage['tp_buffer'].sample(self.batch_size)

        tp_node_feature, tp_adjacency = self._sample_from_tp_batch(tp_mini_batch)
        tp_loss = self._update_social_regressor(tp_node_feature, tp_adjacency)

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
                  "alpha": self.alpha,
                  "tp_loss": tp_loss}

        losses.update(critic_loss)

        debug = {"entropy": entropy.item(),
                 "log_alpha": self.log_alpha.item()}

        if self.automatic_entropy_tuning:
            losses.update({"alpha_loss": alpha_loss})
            debug.update({"target_entropy": self.target_entropy})
        if self.per:
            debug.update({'beta' : self.beta})

        return losses, debug

    def _set_args(self, args):
        super(GRAIN, self)._set_args(args)

        self.rescale_xy = torch.ones((1,2,1,1)).to(args.device)
        self.rescale_xy[:,0] = args.max_x
        self.rescale_xy[:,1] = args.max_y

        self.hazard = args.hazard
        self.error_order = args.error_order

    def _build_network(self, state_space, action_space, args):
        super(GRAIN, self)._build_network(state_space, action_space, args)
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
        parameters += list(self.base.parameters())
        self.critic_optim = Adam(parameters, lr=args.lr)

        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

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

        node_feature, ori_node_feature = self._preprocess(node_feature)
        ori_last_loc = ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()
        next_node_feature, next_ori_node_feature = self._preprocess(next_node_feature)
        next_ori_last_loc = next_ori_node_feature[:,:2,-1,:].permute(0, 2, 1).contiguous()

        dynamic_feature = self.gse(node_feature, adjacency, ret_o=True)
        next_dynamic_feature = self.gse(next_node_feature, next_adjacency, ret_o=True)

        state = self.base(kinetic_feature, dynamic_feature, ori_last_loc)
        with torch.no_grad():
            next_state = self.base(next_kinetic_feature, next_dynamic_feature, ori_last_loc=next_ori_last_loc)

        return state, action, reward, mask, next_state

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

    def _tp_regressor(self, storage):
        loss_tp = []
        for iteration, (node_feature, adjacency, gt_node_feature) in enumerate(storage['loader']):

            node_feature = node_feature.to(self.device) # (N, C, T, V)
            adjacency = adjacency.to(self.device) #(N, L, V, V)
            gt_node_feature = gt_node_feature.to(self.device)

            input_data, _ = self._preprocess(node_feature) # (N, 10, T, V)
            loc_GT, _ = self._preprocess(gt_node_feature)
            loc_GT = loc_GT[:,:2,...]

            output_mask = input_data[:,-1:,:,:] # (N, 1, T, V)
            last_loc = self.gse.reshape_for_lstm(input_data[:,:2])[:,-1:,:] # (N*V, 1, 2)
            enc_hidden = self.gse(input_data, adjacency)

            predicted = self.tp(last_loc, enc_hidden, self.gse.reshape_for_lstm(loc_GT))

            tp_loss = self._prediction_error(predicted, loc_GT, output_mask, error_order = 1)
            loss_tp.append(tp_loss.item())

            self.tp_optim.zero_grad()
            tp_loss.backward()
            nn.utils.clip_grad_norm_(
                list(itertools.chain(*[g['params'] for g in self.tp_optim.param_groups])),
                self.max_grad_norm)
            self.tp_optim.step()

        return np.mean(loss_tp)

    def reset(self):
        pass

    def get_optimizer(self):
        optimizer = {}
        optimizer.update(super(GRAIN, self).get_optimizer())
        optimizer.update({'tp': self.tp_optim})
        return optimizer

    def vis_log(self, state):
        nodes=state['graph_feature']
        f_adj=state['adjacency']
        d_adj=[p.data.detach().cpu().numpy() for p in self.gse.edge_importance]
        p_traj=self.predicted
        p_mask=self.predicted_mask

        # Plot Trajectory Pediction Result
        color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        num_obj = int(p_mask.sum())
        objs = np.stack(nodes).transpose((2,1,0))[:num_obj,:,:2]
        obj_hists = objs.reshape(-1,2)
        obj_preds = np.stack(p_traj[0]).transpose((2,1,0))[:num_obj].reshape(-1,2)
        obj_locs = objs[:,-1]

        ax = self.ax1
        ax.scatter(0, 0, c=color_list[0], marker='*', s=300, alpha=0.5, label='ego')
        ax.scatter(obj_locs[1:,0], obj_locs[1:,1], c=color_list[1], marker='s', s=150, alpha=0.5, label='others')
        ax.scatter(obj_hists[:, 0], obj_hists[:, 1], c=color_list[2], alpha=0.4, label='history')
        ax.scatter(obj_preds[:, 0], obj_preds[:, 1], c=color_list[3], alpha=0.4, label='prediction')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.tick_params('x',labelsize=16)
        ax.tick_params('y',labelsize=16)
        ax.set_xlabel('x', fontsize=18)
        ax.set_ylabel('y', fontsize=18)
        ax.set_title('Trajectory Prediction ', fontsize=20)
        ax.axis('on')
        ax.axis('auto')
        self.fig1.tight_layout()
        self.camera1.snap()

        # Plot Fixed Graph
        ax = self.ax2
        pos = {ind : loc for ind, loc in enumerate(obj_locs)}
        G = nx.Graph()
        G.add_nodes_from(pos.keys())
        for i, p in pos.items():
            G.nodes[i]['pos'] = p

        nodelist = list(pos.keys())
        nx.draw_networkx_nodes(G,
                            pos=nx.get_node_attributes(G,'pos'),
                            nodelist=[nodelist[0]],
                            node_size=500, node_color=color_list[0],
                            node_shape='*',
                            alpha=0.6,
                            label='ego',
                            ax=ax)

        nx.draw_networkx_nodes(G,
                            pos=nx.get_node_attributes(G,'pos'),
                            nodelist=nodelist[1:],
                            node_size=400, node_color=color_list[1],
                            alpha=0.6,
                            label='others',
                            ax=ax)

        for i, adj in enumerate(f_adj):
            edges = nx.from_numpy_matrix(adj[:num_obj][:,:num_obj]).edges()
            nx.draw_networkx_edges(G,
                                pos=nx.get_node_attributes(G,'pos'),
                                edgelist=edges,
                                width=3.0,
                                alpha=np.exp(-(i)),
                                edge_cmap=plt.cm.Greys ,
                                style='dashed',
                                ax=ax)

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.tick_params('x',labelsize=16)
        ax.tick_params('y',labelsize=16)
        ax.set_xlabel('x', fontsize=18)
        ax.set_ylabel('y', fontsize=18)
        ax.set_title('Fixed Graph', fontsize=18)
        ax.axis('on')
        ax.axis('auto')
        self.fig2.tight_layout()
        self.camera2.snap()

        # Plot Dynamic Graph
        G = nx.Graph()
        G.add_nodes_from(pos.keys())
        for i, p in pos.items():
            G.nodes[i]['pos'] = p

        nodelist = list(pos.keys())
        for layer, ax in enumerate(self.axes):
            nx.draw_networkx_nodes(G,
                                pos=nx.get_node_attributes(G,'pos'),
                                nodelist=[nodelist[0]],
                                node_size=500, node_color=color_list[0],
                                node_shape='*',
                                alpha=0.6,
                                label='ego',
                                ax=ax)

            nx.draw_networkx_nodes(G,
                                pos=nx.get_node_attributes(G,'pos'),
                                nodelist=nodelist[1:],
                                node_size=400, node_color=color_list[1],
                                alpha=0.6,
                                label='others',
                                ax=ax)

            for hop, adj in enumerate(d_adj[layer]):
                thres = (adj.max() + adj.min()) / 2
                adj[adj <= thres] = 0.
                edges = nx.from_numpy_matrix(adj[:num_obj][:,:num_obj]).edges()
                nx.draw_networkx_edges(G,
                                    pos=nx.get_node_attributes(G,'pos'),
                                    edgelist=edges,
                                    width=3.0,
                                    alpha=np.exp(-(hop)),
                                    edge_cmap=plt.cm.Greys ,
                                    style='dashed',
                                    ax=ax)

            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            ax.tick_params('x',labelsize=16)
            ax.tick_params('y',labelsize=16)
            ax.set_xlabel('x', fontsize=18)
            ax.set_ylabel('y', fontsize=18)
            ax.set_title('{} th Layer '.format(layer + 1), fontsize=18)
            ax.axis('on')
            ax.axis('auto')


        self.fig3.suptitle('Dynamic Graph', fontsize=20)
        self.fig3.tight_layout()
        self.camera3.snap()

    def set_camera(self):
        plt.close('all')
        self.fig1, self.ax1 = plt.subplots()
        self.camera1 = Camera(self.fig1)
        self.fig2, self.ax2 = plt.subplots()
        self.camera2 = Camera(self.fig2)
        self.fig3, self.axes = plt.subplots(1,4, figsize=(5*4,5))
        self.camera3 = Camera(self.fig3)
        plt.ion()

    def save_video(self, args):
        file_dir = os.path.join(args.output, 'video')
        os.makedirs(file_dir, exist_ok=True)

        file_name1 = '{}_trajectory_prediction.mp4'.format(self.total_it)
        file_path1 = os.path.join(file_dir, file_name1)
        animation = self.camera1.animate(interval=20, blit=True)
        animation.save(file_path1)
        prCyan('\t{} is saved'.format(file_path1))

        file_name2 = '{}_fixed_graph.mp4'.format(self.total_it)
        file_path2 = os.path.join(file_dir, file_name2)
        animation = self.camera2.animate(interval=20, blit=True)
        animation.save(file_path2)
        prCyan('\t{} is saved'.format(file_path2))

        file_name3 = '{}_dynamic_graph.mp4'.format(self.total_it)
        file_path3 = os.path.join(file_dir, file_name3)
        animation = self.camera3.animate(interval=20, blit=True)
        animation.save(file_path3)
        prCyan('\t{} is saved'.format(file_path3))

        self.set_camera()

        if args.write_summary and args.wandb:
            wandb.log({"dynamic_graph": wandb.Video(file_path3)}, step=self.total_it)
            wandb.log({"trajectory_prediction": wandb.Video(file_path1)}, step=self.total_it)
            wandb.log({"fixed_graph": wandb.Video(file_path2)}, step=self.total_it)
