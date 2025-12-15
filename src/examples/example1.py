import sys, pathlib
import numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # ensure src on path
import torch
from gda import run_gda_solve

def obj_func(x):
    x = x.to(torch.float64)
    x1, x2 = x[0], x[1]
    return (x1**2 + x2**2 +3 ) / (1 + 2 * x1 + 8 * x2)

def grad_func(x):
    # compute gradient via autograd
    xv = x.clone().detach().requires_grad_(True)
    f = obj_func(xv)
    f.backward()
    return xv.grad.detach()

def proj_func(x):
    """
    Projects x onto C: {x >= 0, x1^2 + 2x1x2 >= 4}.
    Uses an iterative correction method for the nonlinear constraint.
    """
    with torch.no_grad():
        x = x.to(torch.float64)

        # 1. Enforce non-negativity first
        x = torch.maximum(x, torch.tensor(0.0, dtype=torch.float64))

        # 2. Enforce nonlinear constraint: c(x) = x1^2 + 2*x1*x2 - 4 >= 0
        # We perform a few steps of projection if violated
        for _ in range(20):  # Iterative projection
            x1, x2 = x[0], x[1]
            constraint_val = x1 ** 2 + 2 * x1 * x2 - 4

            if constraint_val >= -1e-6:  # Satisfied (with tolerance)
                break

            # Calculate gradient of the constraint boundary
            # grad_c = [2x1 + 2x2, 2x1]
            grad_c = torch.tensor([2 * x1 + 2 * x2, 2 * x1], dtype=torch.float64)
            norm_grad_sq = torch.sum(grad_c ** 2)

            if norm_grad_sq < 1e-6:
                break  # Avoid division by zero

            # Move x towards the feasible region (Newton-like update)
            # delta = - (val / |grad|^2) * grad
            step = - (constraint_val / norm_grad_sq) * grad_c
            x = x + step

            # Re-enforce non-negativity
            x = torch.maximum(x, torch.tensor(0.0, dtype=torch.float64))
    return x


def run_example():
    # Feasible starting point, but far from optimum
    x0 = torch.tensor([3.0, 2.0], dtype=torch.float64)
    sigma = 0.1        
    kappa = 0.5         

    x_opt = run_gda_solve(grad_func, proj_func, obj_func, x0, step_size=0.1, sigma=sigma, kappa=kappa, max_iter=2000, tol=1e-8)
    print("x_opt =", x_opt)
    print("f(x_opt)", obj_func(x_opt))
    return x_opt

if __name__ == "__main__":
    run_example()