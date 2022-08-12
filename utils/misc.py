import os

import random
import numpy as np
import torch
import torch.nn as nn
import envs
from torch.autograd import grad

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))


def seed_torch(seed=0, cuda=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_output_folder(p_dir, project, algo, env, seed, study):
    if project is not None:
        c_dir = os.path.join(p_dir, project, env, algo)
    else:
        c_dir = os.path.join(p_dir, env, algo)
    if study is not None:
        c_dir = os.path.join(c_dir, study)
    if seed > 0:
        c_dir = os.path.join(c_dir, str(seed))
    os.makedirs(c_dir, exist_ok=True)
    return c_dir


def get_normalize_rms(env):
    if isinstance(env, envs.ObservationNormalize):
        return getattr(env, 'ob_rms', None)
    elif isinstance(env, envs.VecNormalize):
        return getattr(env, 'ob_rms', None)
    elif hasattr(env, 'venv'):
        return get_normalize_rms(env.venv)
    elif hasattr(env, 'env'):
        return get_normalize_rms(env.env)
    return None

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def xavier_weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def he_weights_init_(m, nonlinearity='relu'):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
        torch.nn.init.constant_(m.bias, 0)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, end_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - ((initial_lr - end_lr) * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_params(parameterized_fun, new_params):
    '''
    Set the parameters of parameterized_fun to new_params

    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator to be updated

    update : torch.FloatTensor
        a flattened version of the parameters to be set
    '''
    index = 0
    for params in parameterized_fun.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length

def flatten(vecs):
    '''
    Return an unrolled, concatenated copy of vecs

    Parameters
    ----------
    vecs : list
        a list of Pytorch Tensor objects

    Returns
    -------
    flattened : torch.FloatTensor
        the flattened version of vecs
    '''

    flattened = torch.cat([v.view(-1) for v in vecs])

    return flattened

def flat_grad(functional_output, inputs, retain_graph=False, create_graph=False):
    '''
    Return a flattened view of the gradients of functional_output w.r.t. inputs

    Parameters
    ----------
    functional_output : torch.FloatTensor
        The output of the function for which the gradient is to be calculated

    inputs : torch.FloatTensor (with requires_grad=True)
        the variables w.r.t. which the gradient will be computed

    retain_graph : bool
        whether to keep the computational graph in memory after computing the
        gradient (not required if create_graph is True)

    create_graph : bool
        whether to create a computational graph of the gradient computation
        itself

    Return
    ------
    flat_grads : torch.FloatTensor
        a flattened view of the gradients of functional_output w.r.t. inputs
    '''

    if create_graph == True:
        retain_graph = True

    grads = grad(functional_output, inputs, retain_graph=retain_graph, create_graph=create_graph)
    flat_grads = flatten(grads)

    return flat_grads

def flat_hessian(grads, inputs, retain_graph=False, create_graph=False):

    if create_graph == True:
        retain_graph = True

    vecs = grad(grads, inputs, retain_graph=retain_graph, create_graph=create_graph)
    hessians_flatten = torch.cat([v.contiguous().view(-1) for v in vecs]).data

    return hessians_flatten

def get_flat_params(parameterized_fun):
    '''
    Get a flattened view of the parameters of a function approximator

    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator for which the parameters are to be returned

    Returns
    -------
    flat_params : torch.FloatTensor
        a flattened view of the parameters of parameterized_fun
    '''
    parameters = parameterized_fun.parameters()
    flat_params = flatten([param.data.view(-1) for param in parameters])

    return flat_params

def in_elipsoid(pos, coeff):
    return ((pos[:,0] - coeff[2])** 2 / coeff[0]**2) + (pos[:,1] ** 2 / coeff[1] ** 2) <=  1

def in_rectangle(pred, length, width):
    return [(- length <= pos[0] <= length) and (- width <= pos[1] <= width) for pos in pred]