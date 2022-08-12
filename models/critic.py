import torch
from torch import nn

from utils import he_weights_init_
from layers.build_model import build_mlp


class TwinnedStateFunction(nn.Module):
    def __init__(self, input_dim,
                 output_dim=1,
                 hidden_units=(64, 64),
                 hidden_activation=nn.ReLU()):
        super().__init__()

        self.net1 = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.net2 = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.apply(he_weights_init_)

    def forward(self, states):
        return self.net1(states), self.net2(states)

    def q1(self, states):
        return self.net1(states)

class TwinnedStateActionFunction(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_units=(64, 64),
                 hidden_activation=nn.ReLU()):
        super().__init__()

        self.net1 = build_mlp(
            input_dim=input_dim + action_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.net2 = build_mlp(
            input_dim=input_dim + action_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.apply(he_weights_init_)

    def forward(self, states, actions):
        xs = torch.cat([states, actions], dim=-1)
        return self.net1(xs), self.net2(xs)

    def q1(self, states, actions):
        return self.net1(torch.cat([states, actions], dim=-1))

class StateActionFunction(nn.Module):
    def __init__(self, input_dim, action_dim, output_dim=1, hidden_units=(64, 64),
                 hidden_activation=nn.ReLU(), output_activation=None):
        super(StateActionFunction, self).__init__()
        self.net = build_mlp(input_dim=input_dim + action_dim,
                             output_dim=output_dim,
                             hidden_units=hidden_units,
                             hidden_activation=hidden_activation,
                             output_activation=output_activation
                             )
        self.apply(he_weights_init_)

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

class StateValueFunction(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_units=(64, 64),
                 hidden_activation=nn.ReLU(), output_activation=None):
        super(StateValueFunction, self).__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )
        self.apply(he_weights_init_)

    def forward(self, states):
        return self.net(states)
