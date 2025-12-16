import torch


class Example1Problem:
    def __init__(self):
        """
        Setup for Example 1 .
        Minimize f(x) = (x1^2 + x2^2 + 3) / (1 + 2x1 + 8x2)
        Subject to: x1^2 + 2x1x2 >= 4  (Derived from -x1^2 - 2x1x2 <= -4)
                    x1, x2 >= 0
        """
        # Optimal solution mentioned in paper
        self.optimal_solution = torch.tensor([0.8922, 1.7957], dtype=torch.float64)

        # Lipschitz constant is not explicitly given for Ex1, so we choose a safe step size.
        # Users can tune this manually.
        self.manual_step_size = 0.05

        print(f"-> Problem Initialized (Example 1)")
        print(f"-> Known Optimal Solution ~ {self.optimal_solution.numpy()}")

    def f_value(self, x):
        """
        Objective function f(x)
        """
        x = x.to(torch.float64)
        x1, x2 = x[0], x[1]

        numerator = x1 ** 2 + x2 ** 2 + 3
        denominator = 1 + 2 * x1 + 8 * x2

        return numerator / denominator

    def get_gradient_autodiff(self, x_val):
        """
        Computes gradient via AutoDiff.
        """
        x_tensor = x_val.clone().detach().requires_grad_(True)
        y = self.f_value(x_tensor)
        y.backward()
        return x_tensor.grad.detach()

    def projection_C(self, x):
        """
        Projects x onto C: {x >= 0, x1^2 + 2x1x2 >= 4}.
        Uses an iterative correction method for the nonlinear constraint.
        """
        with torch.no_grad():
            x = x.to(torch.float64)

            # 1. Enforce non-negativity first
            x = torch.maximum(x, torch.tensor(0.0, dtype=torch.float64))

            # 2. Enforce nonlinear constraint: c(x) = x1^2 + 2*x1*x2 - 4 >= 0
            # We perform a few steps of projection if violated
            for _ in range(10):  # Iterative projection
                x1, x2 = x[0], x[1]
                constraint_val = x1 ** 2 + 2 * x1 * x2 - 4

                if constraint_val >= -1e-6:  # Satisfied (with tolerance)
                    break

                # Calculate gradient of the constraint boundary
                # grad_c = [2x1 + 2x2, 2x1]
                grad_c = torch.tensor([2 * x1 + 2 * x2, 2 * x1], dtype=torch.float64)
                norm_grad_sq = torch.sum(grad_c ** 2)

                if norm_grad_sq < 1e-6:
                    break  # Avoid division by zero

                # Move x towards the feasible region (Newton-like update)
                # delta = - (val / |grad|^2) * grad
                step = - (constraint_val / norm_grad_sq) * grad_c
                x = x + step

                # Re-enforce non-negativity
                x = torch.maximum(x, torch.tensor(0.0, dtype=torch.float64))

            return x

    def get_step_size(self):
        """Returns manual step size since L is not calculated."""
        return self.manual_step_size