import copy
import numpy as np
import torch
import torch.nn as nn

from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import EncoderRNN

class GraphSocialEncoder(nn.Module):
    def __init__(self, in_channels, max_hop, num_node, edge_importance_weighting,
                 temporal_kernel_size=5, graph_hidden_size=64, rnn_hidden_size=64, rnn_num_layer=2):
        super(GraphSocialEncoder, self).__init__()


        A = np.ones((max_hop + 1, num_node, num_node))
        spatial_kernel_size = np.shape(A)[0]
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.st_gcn_networks = nn.ModuleList((
            # nn.BatchNorm2d(in_channels),
            Graph_Conv_Block(in_channels, graph_hidden_size, kernel_size, 1, residual=True),
            Graph_Conv_Block(graph_hidden_size, graph_hidden_size, kernel_size, 1),
            Graph_Conv_Block(graph_hidden_size, graph_hidden_size, kernel_size, 1),
        ))

        # initialize parameters for edge importance weighting (trainable graph)
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.encoder = EncoderRNN(graph_hidden_size, rnn_hidden_size, rnn_num_layer)

        self.num_node = num_node

    def reshape_for_lstm(self, feature):
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

    def reshape_from_lstm(self, feature):
        NV, H = feature.size()
        now_feat = feature.view(-1, self.num_node, H) # (N, V, H)
        return now_feat

    def forward(self, x, adj, ret_h=False, ret_o=False):
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            if type(gcn) is nn.BatchNorm2d:
                x = gcn(x)
            else:
                x, _ = gcn(x, adj + importance)

        x = self.reshape_for_lstm(x) # graph_conv_feature : (N, C, T, V) -> (N*V, T, C)
        enc_output, h = self.encoder(x) # (N*V, T, H), (L, N*V, H)

        # TODO: attention from encoded output sequence
        # use last layer of RNN
        if ret_o:
            # h = h.detach()
            o = self.reshape_from_lstm(h[-1]) # (N*V, H) -> (N, V, H)
            if ret_h:
                return o, h
            else:
                return o
        else:
            return h
