import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add PCD source path
pcd_src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if pcd_src_path not in sys.path:
    sys.path.insert(0, pcd_src_path)

# Add GDA source path
gda_src_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'GDA', 'src'))
if gda_src_path not in sys.path:
    sys.path.insert(0, gda_src_path)

# --- Imports ---
try:
    from algorithms.pcd import run_pcd
    from problems.math_example1 import NonConvexProblem
    from gda_algorithm import GDAOptimizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def main():
    # --- Setup Problem ---
    problem = NonConvexProblem()
    x0 = np.array([0.5, 3.0])
    max_iter = 200
    log_interval = 20  # Print log every 20 iterations

    print(f"--- COMPARISON EXPERIMENT: PCD vs GDA ---")
    print(f"Problem: {problem.name}")
    print(f"Initial Point: {x0}")
    print(f"Max Iterations: {max_iter}")

    # ==========================================
    # 1. Run PCD (Projected Coordinate Descent)
    # ==========================================
    print("\n" + "=" * 60)
    print("1. RUNNING PCD (Fixed Learning Rate = 0.1)")
    print("=" * 60)

    start_time = time.time()
    pcd_trajectory = run_pcd(
        gradient_func=problem.calculate_gradient,
        projection_func=problem.project_to_constraint,
        initial_point=x0,
        learning_rate=0.1,
        max_steps=max_iter,
        random_selection=False
    )
    pcd_time = time.time() - start_time

    # --- PCD Logging ---
    print(f"{'Iter':<6} | {'x1':<10} | {'x2':<10} | {'Objective':<12}")
    print("-" * 50)

    pcd_values = []
    for i, x in enumerate(pcd_trajectory):
        val = problem.calculate_objective(x)
        pcd_values.append(val)

        if i % log_interval == 0 or i == len(pcd_trajectory) - 1:
            print(f"{i:<6} | {x[0]:.6f}   | {x[1]:.6f}   | {val:.6f}")

    print(f"\n-> PCD Finished: f(x*)={pcd_values[-1]:.6f}, Time={pcd_time:.4f}s")

    # ==========================================
    # 2. Run GDA (Gradient Descent Adaptive)
    # ==========================================
    print("\n" + "=" * 60)
    print("2. RUNNING GDA (Adaptive Step Size)")
    print("=" * 60)

    gda = GDAOptimizer(
        func=problem.calculate_objective,
        grad_func=problem.calculate_gradient,
        x0=x0,
        projection_func=problem.project_to_constraint,
        max_iter=max_iter,
        lambda_0=0.1,  # Start with same LR as PCD
        sigma=0.1,
        kappa=0.5,
        verbose=False  # Turn off internal print to use our custom table
    )

    gda_result = gda.optimize()
    gda_values = gda_result['history']['f']
    gda_history_x = gda_result['history']['x']
    gda_step_sizes = gda_result['history']['step_size']

    # --- GDA Logging ---
    print(f"{'Iter':<6} | {'x1':<10} | {'x2':<10} | {'Objective':<12} | {'StepSize (λ)':<12}")
    print("-" * 65)

    for i in range(len(gda_history_x)):
        if i % log_interval == 0 or i == len(gda_history_x) - 1:
            x = gda_history_x[i]
            val = gda_values[i]
            step = gda_step_sizes[i]
            print(f"{i:<6} | {x[0]:.6f}   | {x[1]:.6f}   | {val:.6f}   | {step:.6f}")

    print(f"\n-> GDA Finished: f(x*)={gda_result['f']:.6f}, Time={gda_result['time']:.4f}s")

    # ==========================================
    # 3. Visualization
    # ==========================================
    print("\nGenerating Comparison Chart...")

    plt.figure(figsize=(12, 7))

    # Plot PCD
    plt.plot(range(len(pcd_values)), pcd_values,
             'r-', label=f'PCD (Fixed LR=0.1)', linewidth=2.0, alpha=0.8)

    # Plot GDA
    plt.plot(range(len(gda_values)), gda_values,
             'b--', label=f'GDA (Adaptive)', linewidth=2.0, alpha=0.8)

    # Target optimal
    plt.axhline(y=0.4094, color='k', linestyle=':', alpha=0.6, label='Reference Optimal (0.4094)')

    plt.title(f'Comparison: PCD vs GDA on {problem.name}')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value f(x)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save plot
    output_path = os.path.join(current_dir, 'comparison_pcd_gda_detailed.png')
    plt.savefig(output_path)
    print(f"Chart saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()