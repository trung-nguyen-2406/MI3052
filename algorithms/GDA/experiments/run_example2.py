"""
Example 2: Nonsmooth pseudoconvex optimization with nonconvex inequality constraints
minimize f(x) = e^(x2-3) - 30 / (x1^2 + x3^2 + 2*x2^2 + 4)
subject to constraints
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.gda_algorithm import GDAOptimizer
from src.examples import create_example


def run_example_2():
    """Run Example 2 individually"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: NONSMOOTH PSEUDOCONVEX OPTIMIZATION")
    print("="*80)
    print("""
Problem Definition:
    minimize f(x) = e^(x2-3) - 30 / (x1^2 + x3^2 + 2*x2^2 + 4)
    
    subject to:
        g1(x) = (x1 + x3)^2 + 2*x2^2 ≤ 10
        g2(x) = (x2 - 1)^2 ≤ 1
        2*x1 + 4*x2 + x3 = -1
    
Note: The objective function is nonsmooth pseudoconvex on the feasible region.
    """)
    
    # Create example
    func, grad, proj, x0, name = create_example(2)
    
    print(f"Initial point: x0 = {x0}")
    print(f"Initial function value: f(x0) = {func(x0):.6f}\n")
    
    # Run GDA
    print("=" * 80)
    print("RUNNING ALGORITHM GDA (PROPOSED)")
    print("=" * 80)
    
    gda = GDAOptimizer(
        func=func,
        grad_func=grad,
        x0=x0,
        projection_func=proj,
        max_iter=10000,
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
    run_example_2()
