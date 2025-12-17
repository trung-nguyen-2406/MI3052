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
    from gda_algorithm import GDAOptimizer
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
    gda_lambda = 1.0
    gda_sigma = 1e-4

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

    # --- 4. Visualization (Dual Charts) ---
    # Create a figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Chart 1: Convergence Comparison ---
    ax1.plot(pcd_vals, 'r-', label='PCD Algorithm', linewidth=2)
    ax1.plot(gda_vals, 'b--', label='GDA Algorithm', linewidth=2)
    ax1.axhline(y=0.4094, color='gray', linestyle=':', alpha=0.5, label='Optimal (Paper)')

    ax1.set_title(f'Convergence Comparison (f(x) over Iterations)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylim(0.40, 0.48)  # Focus on convergence zone

    # --- Chart 2: Execution Time Comparison ---
    methods = ['PCD', 'GDA']
    times = [pcd_time, gda_time]
    colors = ['#ff9999', '#66b3ff']  # Light red and Light blue

    bars = ax2.bar(methods, times, color=colors, alpha=0.8, width=0.5)

    ax2.set_title('Execution Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_ylim(0, max(times) * 1.2)  # Add some headroom for labels

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}s',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Layout adjustments
    plt.tight_layout()

    out_path = os.path.join(current_dir, 'final_comparison_with_time.png')
    plt.savefig(out_path)
    print(f"\nChart saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()