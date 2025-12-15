import numpy as np
from scipy.optimize import minimize


class NonConvexProblem:
    def __init__(self):
        self.name = "Nonconvex Example 1"

    def calculate_objective(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        numerator = x1 ** 2 + x2 ** 2 + 3
        denominator = 1 + 2 * x1 + 8 * x2
        return numerator / denominator

    def calculate_gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[0], x[1]

        u = x1 ** 2 + x2 ** 2 + 3
        v = 1 + 2 * x1 + 8 * x2

        du_dx1 = 2 * x1
        du_dx2 = 2 * x2
        dv_dx1 = 2
        dv_dx2 = 8

        # Quotient rule: (u'v - uv') / v^2
        grad_x1 = (du_dx1 * v - u * dv_dx1) / (v ** 2)
        grad_x2 = (du_dx2 * v - u * dv_dx2) / (v ** 2)

        return np.array([grad_x1, grad_x2])

    def project_to_constraint(self, x_current: np.ndarray) -> np.ndarray:
        # Enforce non-negativity first
        x_clean = np.maximum(x_current, 0.0)
        x1, x2 = x_clean[0], x_clean[1]

        # Check constraint: x1^2 + 2x1x2 >= 4
        constraint_val = x1 ** 2 + 2 * x1 * x2 - 4
        if constraint_val >= 0:
            return x_clean

        # --- Optimization for Projection ---
        # Min ||y - x||^2 subject to g(y) >= 0
        def distance_squared(y):
            return np.sum((y - x_clean) ** 2)

        constraints = ({'type': 'ineq', 'fun': lambda y: y[0] ** 2 + 2 * y[0] * y[1] - 4})
        bounds = ((0, None), (0, None))

        result = minimize(
            distance_squared,
            x_clean,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x