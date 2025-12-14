import torch
import time


def run_gd_solver(grad_func, proj_func, obj_func, x0, step_size, max_iter=1000, tol=1e-6):
    """
    Algorithm 2: Gradient Descent Algorithm (GD).

    Inputs:
        step_size: Fixed lambda (Should be 1/L for Example 3).
    """
    x = x0.clone().detach()

    print(f"{'Iter':<10} | {'Objective f(x)':<20} | {'Step Size':<15}")
    print("-" * 50)

    start_time = time.time()

    for k in range(max_iter):
        # 1. Compute Gradient
        grad = grad_func(x)

        # 2. Descent Step: x - lambda * grad
        x_new = x - step_size * grad

        # 3. Projection Step: P_C(...)
        x_new = proj_func(x_new)

        # Convergence Check
        diff = torch.norm(x_new - x)
        val = obj_func(x_new)

        # Log info
        if k % 10 == 0 or k == max_iter - 1:
            print(f"{k:<10} | {val.item():<20.6f} | {step_size:<15.6f}")

        if diff < tol:
            print(f"\n-> Converged at iteration {k}")
            x = x_new
            break

        # Update
        x = x_new

    end_time = time.time()
    print("-" * 50)
    print(f"Total Time: {end_time - start_time:.4f} seconds")
    return x