import torch
import numpy as np
import cvxopt

EPS = 1e-12

cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1e-3
cvxopt.solvers.options['reltol'] = 1e-3
cvxopt.solvers.options['feastol'] = 1e-3
cvxopt.solvers.options['maxiters'] = 50


def quadprog(P, q, G, h, A, b):
    P = cvxopt.matrix(P.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
    return np.array(sol['x'])


def setup_qp_and_solve_for_mgdaplus_1(vec, epsilon, lambda0):

    P = np.dot(vec, vec.T)

    n = P.shape[0]

    q = np.zeros((n, 1))

    A = np.ones(n).T
    b = np.array([1])

    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    G = np.zeros((2 * n, n))
    for i in range(n):
        G[i][i] = -1
        G[n + i][i] = 1
    h = np.zeros((2 * n, 1))
    for i in range(n):
        h[i] = -lb[i]
        h[n + i] = ub[i]
    sol = quadprog(P, q, G, h, A, b).reshape(-1)

    return sol, 1


def get_d_mgdaplus_d(grads, device, epsilon, lambda0):
    sol, _ = setup_qp_and_solve_for_mgdaplus_1(
        grads.cpu().detach().numpy(), epsilon, lambda0)

    sol = torch.from_numpy(sol).float().to(device)
    d = sol @ grads

    descent_flag = 1
    c = -(grads @ d)
    if not torch.all(c <= 1e-6):
        descent_flag = 0

    return d, sol, descent_flag


def _loss_values_to_tensor(loss, num_vectors, reference):
    if torch.is_tensor(loss):
        loss_tensor = loss.to(dtype=reference.dtype, device=reference.device)
    else:
        try:
            loss_iter = iter(loss)
        except TypeError:
            loss_iter = iter([loss])
        values = []
        for item in loss_iter:
            if torch.is_tensor(item):
                values.append(float(item.detach().cpu().item()))
            else:
                values.append(float(item))
        loss_tensor = torch.tensor(values, dtype=reference.dtype, device=reference.device)

    loss_tensor = loss_tensor.reshape(-1)
    if loss_tensor.numel() != num_vectors:
        raise ValueError(f"Expected {num_vectors} loss values, got {loss_tensor.numel()}.")
    return loss_tensor


def Gram_Schmidt(grads, loss, pow):
    num_vectors, _ = grads.shape
    orthogonal = torch.zeros_like(grads, dtype=torch.float64)
    loss_values = _loss_values_to_tensor(loss, num_vectors, orthogonal)

    for i in range(num_vectors):
        vec = grads[i].double().clone()
        numerator = grads[i].double().clone()
        denominator = loss_values[i] ** pow
        for j in range(i):
            projection_denominator = torch.dot(orthogonal[j], orthogonal[j])
            if float(torch.abs(projection_denominator).detach().cpu().item()) <= EPS:
                continue
            ratio = torch.dot(orthogonal[j], vec) / projection_denominator
            numerator -= ratio * orthogonal[j]
            denominator -= ratio
        if float(torch.abs(denominator).detach().cpu().item()) <= EPS:
            denominator = torch.tensor(EPS, dtype=orthogonal.dtype, device=orthogonal.device)
        orthogonal[i] = numerator / denominator
    return orthogonal.to(grads.dtype)


def get_d_adafed(grads):
    norm_sq = torch.norm(grads, dim=1) ** 2
    active = norm_sq > EPS
    d = torch.zeros_like(grads[0])
    if not torch.any(active):
        return d
    active_grads = grads[active]
    active_norm_sq = norm_sq[active]
    total = torch.sum(1.0 / active_norm_sq)
    for grad, grad_norm_sq in zip(active_grads, active_norm_sq):
        lamb = 1.0 / (grad_norm_sq * total)
        d += lamb * grad
    return d
