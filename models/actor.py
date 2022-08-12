import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from utils import xavier_weights_init_, he_weights_init_


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, activation=nn.Tanh()):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        # self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_actions), requires_grad=True)

        self.apply(xavier_weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        self.activation = activation

    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        log_std = self.action_log_std.expand_as(mean)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y = torch.tanh(x)

        log_prob = normal.log_prob(x)
        # Enforcing Action Bound
        # log_prob -= torch.log(1 - y.pow(2) + epsilon)
        # Numverically Stable Enforcing Action Bound
        log_prob -= 2 *(torch.log(2 * torch.ones_like(x)) \
            - x - F.softplus(- 2 * x))
        log_prob = log_prob.sum(1, keepdim=True)

        action = y * self.action_scale + self.action_bias
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def log_prob(self, state, action, mulogstd=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        action = (action - self.action_bias) / self.action_scale
        log_prob = normal.log_prob(atanh(action))
        log_prob -= 2 *(torch.log(2 * torch.ones_like(action)) \
            - action - F.softplus(- 2 * action))
        log_prob = log_prob.sum(1, keepdim=True)
        return (log_prob, mean, log_std) if mulogstd else log_prob


    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class CategoricalPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(CategoricalPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_actions)
        self.apply(he_weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.head(x)

    def act(self, state):
        action_logits = self.forward(state)
        greedy_actions = torch.argmax(action_logits, dim=-1, keepdim=True)
        return greedy_actions

    def sample(self, state):
        action_logits = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1,1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * epsilon
        log_action_probs =  torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

    def to(self, device):
        return super(CategoricalPolicy, self).to(device)


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(xavier_weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)