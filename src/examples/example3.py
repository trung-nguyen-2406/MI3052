"""
Example 3: Large-scale problem
f(x) = a^T*x + α*x^T*x + β/√(1 + β*x^T*x) * e^T*x
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
    # Parameters
    beta = 0.741271
    alpha = 3 * (beta ** 1.5) * np.sqrt(n) + 1
    
    # Random vectors
    a = torch.randn(n)
    e = torch.ones(n)
    
    # Objective function
    def obj_func(x):
        term1 = torch.dot(a, x)
        term2 = alpha * torch.dot(x, x)
        term3 = (beta / torch.sqrt(1 + beta * torch.dot(x, x))) * torch.dot(e, x)
        return term1 + term2 + term3
    
    # Gradient function
    def grad_func(x):
        grad1 = a
        grad2 = 2 * alpha * x
        
        xx = torch.dot(x, x)
        denom = torch.sqrt(1 + beta * xx)
        grad3_part1 = (beta / denom) * e
        grad3_part2 = -(beta**2 * torch.dot(e, x) / (2 * denom**3)) * x
        grad3 = grad3_part1 + grad3_part2
        
        return grad1 + grad2 + grad3
    
    # Projection function: x >= 1
    def proj_func(x):
        return torch.clamp(x, min=1.0)
    
    # Initial point
    x0 = torch.ones(n) + torch.rand(n) * 0.1
    
    # Calculate Lipschitz constant: L = 4β^(3/2)√n + 3α
    L = 4 * (beta ** 1.5) * np.sqrt(n) + 3 * alpha
    
    return obj_func, grad_func, proj_func, x0, L

def run_example_3():
    """Run Example 3 individually"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: LARGE-SCALE PROBLEM")
    print("="*80)
    
    # Create example with different dimensions
    n_values = [10, 20, 50, 100, 200, 500]
    # n_values = [10, 20]
    print(f"""
Problem Definition:
    f(x) = a^T*x + α*x^T*x + β/√(1 + β*x^T*x) * e^T*x
    
    where:
        a = random vector
        e = vector of ones
        β = 0.741271 (parameter)
        α = 3β^(3/2)/√n + 1 (satisfies parameter condition)
        C = {{x ∈ R_+^n : x_i ≥ 1 for all i}}
    
Running Algorithm GDA for different dimensions n.
    """)
    
    print("=" * 80)
    print("COMPUTATIONAL RESULTS FOR EXAMPLE 3")
    print("=" * 80)
    print(f"{'n':<10}{'f(x*)':<15} {'Iterations':<12} {'Time(s)':<12}")
    print("-" * 80)
    
    for n in n_values:
        print(f"{n:<10}", end=" ", flush=True)
        
        # Create problem
        obj_func, grad_func, proj_func, x0, L = create_problem(n)
        
        # Initial step size for GDA: λ₀ = 5/L
        lambda_0 = 5 / L
        
        # Run GDA with history tracking
        start_time = time.time()
        x_final, history = run_gda_solve(
            grad_func=grad_func,
            proj_func=proj_func,
            obj_func=obj_func,
            x0=x0,
            step_size=lambda_0,
            sigma=0.1,
            kappa=0.1,
            max_iter=1000,
            tol=1e-6,
            return_history=True
        )
        elapsed_time = time.time() - start_time
        
        final_obj = float(obj_func(x_final))
        num_iterations = len(history['iterations'])
        
        print(f"{final_obj:<15.4f} {num_iterations:<12} {elapsed_time:<12.4f}")
    
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    run_example_3()