import sys
import os
import torch

# Fix import path if needed
try:
    from gd import run_gd_solver
    from math_example3 import Example3Problem
except ImportError:
    from src.algorithms.gd import run_gd_solver
    from src.problems.math_example3 import Example3Problem

if __name__ == "__main__":
    # Experiment parameters for Algorithm 2
    N_DIM = 50
    MAX_ITER = 1000

    print(f"=== RUNNING ALGORITHM 2 (GD) WITH N={N_DIM} ===")

    problem = Example3Problem(n=N_DIM)

    # Initial solution x0 = (2, ..., 2)
    x0 = torch.full((N_DIM,), 2.0, dtype=torch.float64)

    # Step size 1/L
    step_size = problem.get_step_size()

    # Execute Algorithm 2 (Standard GD)
    optimal_x = run_gd_solver(
        grad_func=problem.get_gradient_autodiff,
        proj_func=problem.projection_C,
        obj_func=problem.f_value,
        x0=x0,
        step_size=step_size,
        max_iter=MAX_ITER
    )

    print("\n=== FINAL RESULTS ===")
    print(f"Optimal f(x): {problem.f_value(optimal_x).item():.6f}")
    print(f"Reference Table 1 (n=50): ~857.1166")
    print(f"Constraint Check (Prod): {torch.prod(optimal_x).item():.6f}")