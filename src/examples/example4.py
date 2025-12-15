"""
Example 4: Gaussian exponential problem
minimize f(x) = -exp(-Σ(xi^2/σi^2))
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import time
sys.path.insert(0, '..')
from gda import run_gda_solve

np.seterr(divide='ignore', invalid='ignore')


def create_problem(n):
    """Create problem instance for dimension n"""
    # Parameters: σ_i = 1.0 for i = 1, ..., n/2; σ_i = 3.0 for i = n/2+1, ..., n
    sigma = torch.ones(n)
    sigma[n//2:] = 3.0
    sigma_sq = sigma ** 2
    
    # Objective function: f(x) = -exp(-Σ(x_i^2/σ_i^2))
    def obj_func(x):
        sum_term = torch.sum(x**2 / sigma_sq)
        return -torch.exp(-sum_term)
    
    # Gradient function: ∇f(x)_i = (2*x_i/σ_i^2) * exp(-Σ(x_j^2/σ_j^2))
    def grad_func(x):
        sum_term = torch.sum(x**2 / sigma_sq)
        exp_term = torch.exp(-sum_term)
        return (2 * x / sigma_sq) * exp_term
    
    # Projection function: depends on constraints (e.g., x >= 0 or box constraints)
    # For now, assume unconstrained or simple box constraints
    def proj_func(x):
        # Example: project onto [0, infinity) or no projection
        return torch.clamp(x, min=0.0)  # x >= 0
    
    # Initial point
    torch.manual_seed(123)
    x0 = torch.rand(n) * 0.5
    
    # Estimate Lipschitz constant
    # For this problem, L depends on the function structure
    # We can use a heuristic value or compute numerically
    L = 2.0 / torch.min(sigma_sq).item()  # Rough estimate
    
    return obj_func, grad_func, proj_func, x0, L


def run_example_4():
    """Run Example 4 individually"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: GAUSSIAN EXPONENTIAL PROBLEM")
    print("="*80)
    
    # Create example with different dimensions
    n_values = [10, 20, 50, 100, 300, 400, 600]
    
    print(f"""
Problem Definition:
    minimize f(x) = -exp(-Σ(x_i^2/σ_i^2))
    
    where:
        σ_i = 1.0 for i = 1, ..., n/2
        σ_i = 3.0 for i = n/2+1, ..., n
        x_i ≥ 0 for all i
    
Running Algorithm GDA for different dimensions n.
    """)
    
    print("=" * 80)
    print("COMPUTATIONAL RESULTS FOR EXAMPLE 4")
    print("=" * 80)
    print(f"{'n':<10} {'-ln(-f(x*))':<15} {'Iterations':<12} {'Time(s)':<12}")
    print("-" * 80)
    
    for n in n_values:
        print(f"{n:<10}", end=" ", flush=True)
        
        # Create problem
        obj_func, grad_func, proj_func, x0, L = create_problem(n)
        
        # Fixed lambda_0
        lambda_0 = 0.5
        
        # Run GDA with history tracking
        start_time = time.time()
        x_final, history = run_gda_solve(
            grad_func=grad_func,
            proj_func=proj_func,
            obj_func=obj_func,
            x0=x0,
            step_size=lambda_0,
            sigma=0.1,
            kappa=0.5,
            max_iter=10,
            tol=1e-12,
            return_history=True
        )
        elapsed_time = time.time() - start_time
        
        final_obj = float(obj_func(x_final))
        num_iterations = len(history['iterations']) - 1  # Exclude initial point
        
        # Display -ln(-f(x*))
        if final_obj < 0:
            display_val = -np.log(-final_obj)
        else:
            display_val = final_obj
        
        print(f"{display_val:<15.4f} {num_iterations:<12} {elapsed_time:<12.4f}")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    run_example_4()
