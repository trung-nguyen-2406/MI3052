"""
Example 1: Simple nonconvex problem
minimize f(x) = (x1^2 + x2^2 + 3) / (1 + 2*x1 + 8*x2)
subject to x ∈ C
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.gda_algorithm import GDAOptimizer
from src.examples import create_example


def run_example_1():
    """Run Example 1 individually"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: SIMPLE NONCONVEX PROBLEM")
    print("="*80)
    print("""
Problem Definition:
    minimize f(x) = (x1^2 + x2^2 + 3) / (1 + 2*x1 + 8*x2)
    subject to x ∈ C where C = {x = (x1, x2) | x1^2 - 2*x1*x2 ≤ -4, x1, x2 ≥ 0}
    """)
    
    # Create example
    func, grad, proj, x0, name = create_example(1)
    
    print(f"Initial point: x0 = {x0}")
    print(f"Initial function value: f(x0) = {func(x0):.6f}\n")
    
    # Run GDA
    print("=" * 80)
    print("RUNNING ALGORITHM GDA (PROPOSED)")
    print("Algorithm GDA: Step-size reduced when sufficient decrease not satisfied")
    print("=" * 80)
    
    gda = GDAOptimizer(
        func=func,
        grad_func=grad,
        x0=x0,
        projection_func=proj,
        max_iter=500,
        tol=1e-6,
        lambda_0=0.1,
        sigma=0.1,
        kappa=0.5,
        verbose=True
    )
    
    gda_result = gda.optimize()
    
    print("\n" + "=" * 80)
    print("GDA RESULTS")
    print("=" * 80)
    print(f"Optimal point: x* = {gda_result['x']}")
    print(f"Optimal value: f(x*) = {gda_result['f']:.6f}")
    print(f"Number of iterations: {gda_result['iterations']}")
    print(f"Computation time: {gda_result['time']:.4f} seconds")
    print(f"Converged: {gda_result['converged']}")
    print()


if __name__ == "__main__":
    run_example_1()
