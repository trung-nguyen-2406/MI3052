"""
Example 3: Large-scale problem
f(x) = a^T*x + α*x^T*x + β/√(1 + β*x^T*x) * e^T*x
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.gda_algorithm import GDAOptimizer
from src.examples import create_example

np.seterr(divide='ignore', invalid='ignore')

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
        C = {{x ∈ R_+^n : 1 ≤ x_i for all i}}
    
Comparing Algorithm GDA vs Algorithm GD for different dimensions n.
    """)
    
    print("=" * 80)
    print("COMPUTATIONAL RESULTS FOR EXAMPLE 3")
    print("=" * 80)
    print(f"{'n':<10}{'GDA f(x*)':<15} {'GDA Iter':<12} {'GDA Time(s)':<12}")
    print("-" * 80)
    
    results_gda = []
    results_gd = []
    
    for n in n_values:
        print(f"{n:<10}", end=" ", flush=True)
        
        # Create example
        func, grad, proj, x0, name = create_example(3, n)
        
        # Calculate Lipschitz constant: L = 4β^(3/2)√n + 3α
        beta = 0.741271
        alpha = 3 * (beta ** 1.5) * np.sqrt(n + 1)
        L = 4 * (beta ** 1.5) * np.sqrt(n) + 3 * alpha
        
        # Initial step size for GDA: λ₀ = 5/L
        lambda_0 = 5 / L
        
        # Run GDA
        # print(f"x0 : {x0}")
        gda = GDAOptimizer(
            func=func,
            grad_func=grad,
            x0=x0,
            projection_func=proj,
            max_iter=1000,
            tol=1e-6,
            lambda_0=lambda_0,
            sigma=0.1,
            kappa=0.1,
            verbose=False
        )
        
        gda_result = gda.optimize()
        results_gda.append(gda_result)
        
        print(f"{gda_result['f']:<15.4f} {gda_result['iterations']:<12} {gda_result['time']:<12.4f}")
    
    print()
    
    # Plot comparison for largest dimension
    # print("=" * 80)
    # print(f"DETAILED RESULTS FOR n = {n_values[-1]}")
    # print("=" * 80)
    
    # func, grad, proj, x0, name = create_example(3, n_values[-1])
    # beta = 0.741271
    # alpha = 3 * (beta ** 1.5) * np.sqrt(n + 1)
    # L = 4 * (beta ** 1.5) * np.sqrt(n) + 3 * alpha
    
    # # Initial step size for GDA: λ₀ = 5/L
    # lambda_0 = 5 / L
    # gda = GDAOptimizer(
    #     func=func,
    #     grad_func=grad,
    #     x0=x0,
    #     projection_func=proj,
    #     max_iter=1000,
    #     tol=1e-6,
    #     lambda_0=lambda_0,
    #     sigma=0.001,
    #     kappa=0.05,
    #     verbose=False
    # )
    
    # gda_result = gda.optimize()
    
    # print(f"\nGDA Results (n={n_values[-1]}):")
    # print(f"  Optimal value: f(x*) = {gda_result['f']:.6f}")
    # print(f"  Iterations: {gda_result['iterations']}")
    # print(f"  Time: {gda_result['time']:.4f} seconds")
    # print()


if __name__ == "__main__":
    run_example_3()
