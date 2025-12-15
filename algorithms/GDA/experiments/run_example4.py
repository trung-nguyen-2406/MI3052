"""
Example 4: Gaussian exponential problem
minimize f(x) = -exp(-Σ(xi^2/σi^2))
subject to constraints
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.gda_algorithm import GDAOptimizer
from src.examples import create_example


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
    
Comparing Algorithm GDA with Algorithm RNN (Recurrent Neural Network)
from Liu et al. (2022) for different dimensions n.
    """)
    
    print("=" * 80)
    print("COMPUTATIONAL RESULTS FOR EXAMPLE 4")
    print("=" * 80)
    print(f"{'n':<10} {'-ln(-f(x*))':<15} {'GDA Iter':<12} {'GDA Time(s)':<12}")
    print("-" * 50)
    
    results_gda = []
    
    for n in n_values:
        print(f"{n:<10}", end=" ", flush=True)
        
        # Create example
        func, grad, proj, x0, name = create_example(4, n)
        
        # Fixed lambda_0 that works well
        lambda_0 = 0.5
        
        # Run GDA with fixed 10 iterations as per paper
        gda = GDAOptimizer(
            func=func,
            grad_func=grad,
            x0=x0,
            projection_func=proj,
            max_iter=10,  # Fixed at 10 as per Table 2
            tol=1e-12,
            lambda_0=lambda_0,
            sigma=0.1,
            kappa=0.5,
            verbose=False
        )
        
        gda_result = gda.optimize()
        results_gda.append(gda_result)
        
        # Display -ln(-f(x*))
        f_val = gda_result['f']
        if f_val < 0:
            display_val = -np.log(-f_val)
        else:
            display_val = f_val
        
        print(f"{display_val:<15.4f} {gda_result['iterations']:<12} {gda_result['time']:<12.4f}")
    
    print()
    
    # Plot comparison for largest dimension
    # print("=" * 80)
    # print(f"DETAILED RESULTS FOR n = {n_values[-1]}")
    # print("=" * 80)
    
    # func, grad, proj, x0, name = create_example(4, n_values[-1])
    
    # gda = GDAOptimizer(
    #     func=func,
    #     grad_func=grad,
    #     x0=x0,
    #     projection_func=proj,
    #     max_iter=1000,
    #     tol=1e-6,
    #     lambda_0=1.0,
    #     sigma=0.1,
    #     kappa=0.5,
    #     verbose=False
    # )
    
    # gda_result = gda.optimize()
    
    # print(f"\nGDA Results (n={n_values[-1]}):")
    # print(f"  Optimal value: f(x*) = {gda_result['f']:.6f}")
    # print(f"  Iterations: {gda_result['iterations']}")
    # print(f"  Time: {gda_result['time']:.4f} seconds")
    # print()


if __name__ == "__main__":
    run_example_4()
