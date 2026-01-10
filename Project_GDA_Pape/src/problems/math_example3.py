import torch
import numpy as np


class Example3Problem:
    def __init__(self, n):
        """
        [cite_start]Initializes problem constants based on Example 3 [cite: 299-307].
        """
        self.n = n
        self.beta = 0.741271

        # Parameter alpha and L derived from beta and n
        self.alpha = 3 * (self.beta ** 1.5) * np.sqrt(n + 1)
        # Lipschitz constant L for Algorithm 2 step size
        self.L = 4 * (self.beta ** 1.5) * np.sqrt(n) + 3 * self.alpha

        self.e = torch.arange(1, n + 1, dtype=torch.float64)

        # Deterministic initialization
        torch.manual_seed(42)
        self.a = torch.rand(n, dtype=torch.float64) + 0.1

        print(f"-> Problem Initialized (n={n})")
        print(f"-> Calculated Lipschitz L = {self.L:.4f}")

    def f_value(self, x):
        """Computes f(x)"""
        x = x.to(torch.float64)
        term1 = torch.dot(self.a, x)
        term2 = self.alpha * torch.dot(x, x)
        dot_ex = torch.dot(self.e, x)
        dot_xx = torch.dot(x, x)
        numerator = self.beta * dot_ex
        denominator = torch.sqrt(1 + self.beta * dot_xx)
        return term1 + term2 + (numerator / denominator)

    def get_gradient_autodiff(self, x_val):
        """Computes gradient via autodiff."""
        x_tensor = x_val.clone().detach().requires_grad_(True)
        y = self.f_value(x_tensor)
        y.backward()
        return x_tensor.grad.detach()

    def projection_C(self, x):
        """
        Exact Euclidean Projection onto C = {z | z > 0, prod(z) >= 1}.
        Solves: min ||z - x||^2 subject to sum(log(z)) >= 0.
        Uses Newton's method to find Lagrange multiplier mu.
        """
        with torch.no_grad():
            x = x.to(torch.float64)

            # Check if x is already feasible (x > 0 AND prod(x) >= 1)
            # Use log sum for numerical stability with large n
            is_positive = torch.all(x > 0)
            if is_positive:
                log_prod = torch.sum(torch.log(x))
                if log_prod >= -1e-7:  # Allow small tolerance
                    return x

            # If not feasible, solve for mu using Newton's method
            # z_i(mu) = (x_i + sqrt(x_i^2 + 4*mu)) / 2
            # We find mu > 0 such that sum(log(z_i)) = 0

            mu = torch.tensor(1.0, dtype=torch.float64, device=x.device)

            for _ in range(20):  # 20 iterations is usually sufficient
                # Compute z and terms
                disc = x ** 2 + 4 * mu
                sqrt_disc = torch.sqrt(disc)
                z = (x + sqrt_disc) / 2.0

                # Function phi(mu) = sum(log(z)) -> Target 0
                phi = torch.sum(torch.log(z))

                # Derivative phi'(mu) = sum(1 / (z * sqrt_disc))
                d_phi = torch.sum(1.0 / (z * sqrt_disc))

                if d_phi < 1e-12:
                    break

                # Newton step
                update = phi / d_phi
                mu = mu - update

                # Enforce constraint mu > 0
                if mu <= 0:
                    mu = torch.tensor(1e-6, dtype=torch.float64, device=x.device)

                # Convergence check
                if torch.abs(update) < 1e-6 and torch.abs(phi) < 1e-6:
                    break

            # Final projection with optimized mu
            z = (x + torch.sqrt(x ** 2 + 4 * mu)) / 2.0
            return z

    def get_step_size(self):
        """Returns step size 1/L for Algorithm 2."""
        return 1.0 / self.L