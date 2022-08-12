import torch

def discount(vals, discount_term):
    n = vals.size(0)
    disc_pows = torch.pow(discount_term, torch.arange(n).float()).to(vals.device)
    reverse_indxs = torch.arange(n - 1, -1, -1)

    discounted = torch.cumsum((vals * disc_pows)[reverse_indxs], dim=-1)[reverse_indxs] / disc_pows

    return discounted

def compute_advs(actual_vals, exp_vals, discount_term, bias_red_param):
    exp_vals_next = torch.cat([exp_vals[1:], torch.tensor([0.0]).to(exp_vals.device)])
    td_res = actual_vals + discount_term * exp_vals_next - exp_vals
    advs = discount(td_res, discount_term * bias_red_param)

    return advs