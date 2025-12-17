import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to local PCD src
pcd_src = os.path.abspath(os.path.join(current_dir, '..', 'src'))
# Path to sibling GDA src
gda_src = os.path.abspath(os.path.join(current_dir, '..', '..', 'GDA', 'src'))

if pcd_src not in sys.path: sys.path.insert(0, pcd_src)
if gda_src not in sys.path: sys.path.insert(0, gda_src)

# --- Imports ---
try:
    from problems.math_example1 import NonConvexProblem
    from algorithms.pcd import run_pcd
    from gda_algorithm import GDAOptimizer  # Import from sibling directory
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'GDA' folder exists parallel to 'PCD' folder.")
    sys.exit(1)


def main():
    # --- 1. Problem Setup ---
    problem = NonConvexProblem()
    x0 = np.array([0.5, 3.0])
    max_iter = 200

    print(f"--- FINAL COMPARISON: PCD vs GDA ---")
    print(f"Problem: {problem.name}")
    print(f"Start Point: {x0}")

    # --- 2. Run PCD (Projected Coordinate Descent) ---
    # Note: PCD works well with LR=0.5 for this specific problem
    pcd_lr = 0.5
    print(f"\nRunning PCD (Fixed LR = {pcd_lr})...")

    start_t = time.time()
    pcd_traj = run_pcd(
        gradient_func=problem.calculate_gradient,
        projection_func=problem.project_to_constraint,
        initial_point=x0,
        learning_rate=pcd_lr,
        max_steps=max_iter
    )
    pcd_time = time.time() - start_t
    pcd_vals = [problem.calculate_objective(x) for x in pcd_traj]

    print(f"-> PCD Done: f(x*)={pcd_vals[-1]:.6f}, Time={pcd_time:.4f}s")

    # --- 3. Run GDA (Gradient Descent Adaptive) ---
    # Constraint: lambda_0 near 1.0 (compliant with paper remarks)
    gda_lambda = 1.0
    gda_sigma = 1e-4  # Standard Armijo condition

    print(f"\nRunning GDA (Adaptive, Start λ={gda_lambda})...")

    gda = GDAOptimizer(
        func=problem.calculate_objective,
        grad_func=problem.calculate_gradient,
        x0=x0,
        projection_func=problem.project_to_constraint,
        max_iter=max_iter,
        lambda_0=gda_lambda,
        sigma=gda_sigma,
        kappa=0.5,
        verbose=False
    )

    start_t = time.time()
    gda_res = gda.optimize()
    gda_time = time.time() - start_t
    gda_vals = gda_res['history']['f']

    print(f"-> GDA Done: f(x*)={gda_vals[-1]:.6f}, Time={gda_time:.4f}s")

    # --- 4. Visualization (Clean Chart) ---
    plt.figure(figsize=(10, 6))

    # Only plotting PCD and GDA as requested
    plt.plot(pcd_vals, 'r-', label='PCD Algorithm', linewidth=2)
    plt.plot(gda_vals, 'b--', label='GDA Algorithm', linewidth=2)

    # Optional: Reference line (can comment out if strictly unwanted)
    plt.axhline(y=0.4094, color='gray', linestyle=':', alpha=0.5, label='Optimal (Paper)')

    plt.title(f'Convergence Comparison: PCD vs GDA')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Focus on the convergence area
    plt.ylim(0.40, 0.48)

    out_path = os.path.join(current_dir, 'final_comparison.png')
    plt.savefig(out_path)
    print(f"\nChart saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()