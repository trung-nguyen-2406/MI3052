import sys
import os
import torch

# Import path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithms.gd import run_gd_solver
from src.problems.math_example1 import Example1Problem

if __name__ == "__main__":
    # Example 1 is 2-dimensional
    MAX_ITER = 2000

    print(f"=== RUNNING EXAMPLE 1 (NONCONVEX) ===")

    # Setup Problem
    problem = Example1Problem()

    # Initialize x0. Paper mentions various initial solutions.
    # Let's pick a valid point, e.g., [2.0, 2.0] -> 2^2 + 2*2*2 = 12 >= 4 (Feasible)
    x0 = torch.tensor([2.0, 2.0], dtype=torch.float64)

    # Run Solver
    optimal_x = run_gd_solver(
        grad_func=problem.get_gradient_autodiff,
        proj_func=problem.projection_C,
        obj_func=problem.f_value,
        x0=x0,
        step_size=problem.get_step_size(),
        max_iter=MAX_ITER
    )

    print("\n=== FINAL RESULTS ===")
    print(f"Found Solution x*: {optimal_x.numpy()}")
    print(f"Optimal Value f(x*): {problem.f_value(optimal_x).item():.6f}")

    # Validate Constraint
    x1, x2 = optimal_x[0], optimal_x[1]
    const_val = x1 ** 2 + 2 * x1 * x2
    print(f"Constraint Check (x1^2 + 2x1x2 >= 4): {const_val.item():.6f}")
    print(f"Paper Optimal Value Ref: ~0.4101")