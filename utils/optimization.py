import torch
from utils.misc import flat_grad

def get_Hvp_fun(functional_output, inputs, damping_coef=0.0):
    '''
    Returns a function that calculates a Hessian-vector product with the Hessian
    of functional_output w.r.t. inputs

    Parameters
    ----------
    functional_output : torch.FloatTensor (with requires_grad=True)
        the output of the function of which the Hessian is calculated

    inputs : torch.FloatTensor
        the inputs w.r.t. which the Hessian is calculated

    damping_coef : float
        the multiple of the identity matrix to be added to the Hessian
    '''

    inputs = list(inputs)
    grad_f = flat_grad(functional_output, inputs, create_graph=True)

    def Hvp_fun(v, retain_graph=True):
        gvp = torch.matmul(grad_f, v)
        Hvp = flat_grad(gvp, inputs, retain_graph=retain_graph)
        Hvp += damping_coef * v

        return Hvp

    return Hvp_fun

def cg_solver(Avp_fun, b, max_iter=10):
    '''
    Finds an approximate solution to a set of linear equations Ax = b

    Parameters
    ----------
    Avp_fun : callable
        a function that right multiplies a matrix A by a vector

    b : torch.FloatTensor
        the right hand term in the set of linear equations Ax = b

    max_iter : int
        the maximum number of iterations (default is 10)

    Returns
    -------
    x : torch.FloatTensor
        the approximate solution to the system of equations defined by Avp_fun
        and b
    '''
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    for i in range(max_iter):
        Avp = Avp_fun(p, retain_graph=True)

        alpha = torch.matmul(r, r) / torch.matmul(p, Avp)
        x += alpha * p

        if i == max_iter - 1:
            return x

        r_new = r - alpha * Avp
        beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
        r = r_new
        p = r + beta * p


def line_search(search_dir, max_step_len, constraints_satisfied, line_search_coef=0.9,
                max_iter=10):
    '''
    Perform a backtracking line search that terminates when constraints_satisfied
    return True and return the calculated step length. Return 0.0 if no step
    length can be found for which constraints_satisfied returns True

    Parameters
    ----------
    search_dir : torch.FloatTensor
        the search direction along which the line search is done

    max_step_len : torch.FloatTensor
        the maximum step length to consider in the line search

    constraints_satisfied : callable
        a function that returns a boolean indicating whether the constraints
        are met by the current step length

    line_search_coef : float
        the proportion by which to reduce the step length after each iteration

    max_iter : int
        the maximum number of backtracks to do before return 0.0

    Returns
    -------
    the maximum step length coefficient for which constraints_satisfied evaluates
    to True
    '''

    step_len = max_step_len / line_search_coef

    for i in range(max_iter):
        step_len *= line_search_coef

        if constraints_satisfied(step_len * search_dir, step_len):
            return step_len

    return 0.0
