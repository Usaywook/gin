import torch
from torch import nn

from utils.misc import he_weights_init_, xavier_weights_init_

class Concatenator(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=nn.ReLU(), distpool=False, sigma=None):
        super(Concatenator, self).__init__()
        input_size = input_dim[1]
        self.dynamic_emb = nn.Sequential(nn.Linear(input_size, hidden_dim), activation)

        if isinstance(activation, nn.Tanh):
            self.apply(xavier_weights_init_)
        elif isinstance(activation, nn.ReLU):
            self.apply(he_weights_init_)

        self.distpool = distpool
        self.sigma = sigma

    def forward(self, kinetic, dynamic, ori_last_loc=None):
        if ori_last_loc is not None:
            dynamic = torch.cat([dynamic, ori_last_loc], dim=-1)# (N, V, H) -> (N, V, H + 2)
        dynamic = self.dynamic_emb(dynamic) # (N, V, H2)

        if self.distpool:
            attention = torch.exp(- torch.linalg.norm(ori_last_loc, dim=-1, keepdim=True) / self.sigma) # (N, V, 1)
            dynamic = torch.mean(dynamic * attention, dim=1) # (N, H2)
        else:
            dynamic = torch.max(dynamic, dim=1)[0]  # (N, H2)
        state = torch.cat([kinetic, dynamic], dim=-1) # (N, H1 + H2)
        return state
