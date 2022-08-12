import torch
import torch.nn as nn
from torch.nn import Linear, Module, Parameter, Sequential, Tanh
from torch.distributions import Independent
from torch.distributions.normal import Normal

def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation:
        layers.append(output_activation)
    return nn.Sequential(*layers)

def conv2d_size_out(size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1

def build_cnn(state_size, cnn_hidden):
    input_size = state_size[0]
    CROPPED_IMAGE_H = state_size[1]
    CROPPED_IMAGE_W = state_size[2]

    convw = conv2d_size_out(conv2d_size_out(CROPPED_IMAGE_H, 8, 4), 4, 2)
    convh = conv2d_size_out(conv2d_size_out(CROPPED_IMAGE_W, 8, 4), 4, 2)
    linear_input_size = convw * convh * 32
    return torch.nn.Sequential(torch.nn.Conv2d(input_size, 16, 8, 4), torch.nn.ReLU(),
                                   torch.nn.Conv2d(16, 32, 4, 2), torch.nn.ReLU(),
                                   torch.nn.Flatten(),
                                   torch.nn.Linear(linear_input_size, cnn_hidden), torch.nn.ReLU())

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class DiagGaussianLayer(Module):
    '''
    Implements a layer that outputs a Gaussian distribution with a diagonal
    covariance matrix

    Attributes
    ----------
    log_std : torch.FloatTensor
        the log square root of the diagonal elements of the covariance matrix

    Methods
    -------
    __call__(mean)
        takes as input a mean vector and outputs a Gaussian distribution with
        diagonal covariance matrix defined by log_std

    '''

    def __init__(self, output_dim=None, log_std=None):
        Module.__init__(self)

        self.log_std = log_std

        if log_std is None:
            self.log_std = Parameter(torch.zeros(output_dim), requires_grad=True)

    def __call__(self, mean):
        # std = torch.clamp(torch.exp(self.log_std), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = torch.exp(self.log_std)
        normal_dist = Independent(Normal(loc=mean, scale=std), 1)

        return normal_dist

def build_layers(input_dim, hidden_dims, output_dim):
    '''
    Returns a list of Linear and Tanh layers with the specified layer sizes

    Parameters
    ----------
    input_dim : int
        the input dimension of the first linear layer

    hidden_dims : list
        a list of type int specifying the sizes of the hidden layers

    output_dim : int
        the output dimension of the final layer in the list

    Returns
    -------
    layers : list
        a list of Linear layers, each one followed by a Tanh layer, excluding the
        final layer
    '''

    layer_sizes = [input_dim] + hidden_dims + [output_dim]
    layers = []

    for i in range(len(layer_sizes) - 1):
        layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], bias=True))

        if i != len(layer_sizes) - 2:
            layers.append(Tanh())

    return layers

def build_diag_gauss_policy(state_dim, hidden_dims, action_dim,
    log_std=None):
    '''
    Build a multilayer perceptron with a DiagGaussianLayer at the output layer

    Parameters
    ----------
    state_dim : int
        the input size of the network

    hidden_dims : list
        a list of type int specifying the sizes of the hidden layers

    action_dim : int
        the dimensionality of the Gaussian distribution to be outputted by the
        policy

    log_std : torch.FloatTensor
        the log square root of the diagonal elements of the covariance matrix
        (will be set to a vector of zeros if none is specified)

    Returns
    -------
    policy : torch.nn.Sequential
        a pytorch sequential model that outputs a Gaussian distribution
    '''

    layers = build_layers(state_dim, hidden_dims, action_dim)
    layers[-1].weight.data *= 0.1
    layers[-1].bias.data *= 0.0
    layers.append(DiagGaussianLayer(action_dim, log_std))
    policy = Sequential(*layers)

    return policy