import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Path Configuration ---
# Ensure priority for local src module
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from algorithms.pcd import run_pcd
from problems.math_example1 import NonConvexProblem


def main():
    # --- Initialization ---
    problem = NonConvexProblem()
    start_point = np.array([0.5, 3.0])
    learning_rate = 0.05
    total_steps = 1000

    print(f"Algorithm: PCD | Problem: {problem.name}")
    print(f"Initial State: {start_point}")

    # --- Execution ---
    trajectory = run_pcd(
        gradient_func=problem.calculate_gradient,
        projection_func=problem.project_to_constraint,
        initial_point=start_point,
        learning_rate=learning_rate,
        max_steps=total_steps,
        random_selection=False
    )

    final_solution = trajectory[-1]
    final_objective = problem.calculate_objective(final_solution)

    # --- Logging Process ---
    print("\n" + "=" * 50)
    print(f"{'Iter':<5} | {'x1':<12} | {'x2':<12} | {'Objective':<12}")
    print("-" * 50)

    for i, x in enumerate(trajectory):
        # Print every 10 steps or the last step
        if i % 10 == 0 or i == len(trajectory) - 1:
            obj_val = problem.calculate_objective(x)
            print(f"{i:<5} | {x[0]:.6f}     | {x[1]:.6f}     | {obj_val:.6f}")

    print("=" * 50 + "\n")

    print(f"Optimized Solution: {final_solution}")
    print(f"Objective Value: {final_objective:.4f}")

    # --- Visualization ---
    x1_values = trajectory[:, 0]
    x2_values = trajectory[:, 1]
    iterations = np.arange(len(trajectory))

    plt.figure(figsize=(10, 6))

    plt.plot(iterations, x1_values, 'r-', label='$x_1(t)$', linewidth=1.5)
    plt.plot(iterations, x2_values, 'g-', label='$x_2(t)$', linewidth=1.5)

    plt.title('Computation Results: Projected Coordinate Descent')
    plt.xlabel('Iteration')
    plt.ylabel('Value x(t)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust view limit
    plt.ylim(0.4, 3.2)

    output_path = os.path.join(current_dir, 'pcd_convergence_chart.png')
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    main()