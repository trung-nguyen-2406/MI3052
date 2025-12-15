import numpy as np
from typing import Callable, List


def run_pcd(
        gradient_func: Callable[[np.ndarray], np.ndarray],
        projection_func: Callable[[np.ndarray], np.ndarray],
        initial_point: np.ndarray,
        learning_rate: float = 0.05,
        max_steps: int = 2000,
        random_selection: bool = False
) -> np.ndarray:
    current_x = initial_point.copy()
    dimensions = len(current_x)
    history = [current_x.copy()]

    for step in range(max_steps):
        # --- Coordinate Selection ---
        if random_selection:
            coord_idx = np.random.randint(dimensions)
        else:
            coord_idx = step % dimensions

        # --- Gradient Calculation ---
        full_gradient = gradient_func(current_x)

        unit_vector = np.zeros(dimensions)
        unit_vector[coord_idx] = 1.0

        # --- Update Step ---
        # x_new = x_old - alpha * grad_i * e_i
        proposal_x = current_x - learning_rate * full_gradient[coord_idx] * unit_vector

        # --- Projection Step ---
        current_x = projection_func(proposal_x)

        history.append(current_x.copy())

    return np.array(history)