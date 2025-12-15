import numpy as np
import time
import sys
import os

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- Problem Imports ---
from problem.math_example_1 import MathExample1
from problem.math_example_2 import MathExample2
from problem.math_example_3 import MathExample3
from problem.math_example_4 import MathExample4

class NeurodynamicRNN:
    """
    Implementation of the Neurodynamic approach using Recurrent Neural Network (RNN)
    Ref: Equations (224) - (238) in the provided document.
    """

    def __init__(self, step_size=1e-3, max_iter=1000):
        self.step_size = step_size
        self.max_iter = max_iter

    def _psi(self, s):
        # Eq. (220): Psi(s) = 1 if s > 0 else 0
        return np.where(s > 0, 1.0, 0.0)

    def _compute_c(self, x, g_val, A, b):
        # Eq. (227) & (228): c(x) calculation
        psi_g = self._psi(g_val)
        
        psi_eq = np.array([])
        if A is not None and b is not None:
            eq_residual = np.abs(np.dot(A, x) - b)
            psi_eq = self._psi(eq_residual)

        all_psi = np.concatenate([psi_g, psi_eq]) if psi_eq.size > 0 else psi_g
        return np.prod(1.0 - all_psi)

    def _subgradient_P(self, g_val, grad_g_val):
        # Eq. (231): Subgradient of inequality penalty P(x)
        active_indices = np.where(g_val > 0)[0]
        
        # Initialize grad_P with correct shape
        grad_P = np.zeros(grad_g_val.shape[1])
        
        if len(active_indices) > 0:
            for idx in active_indices:
                grad_P += grad_g_val[idx]
        
        return grad_P

    def _subgradient_equality(self, x, A, b):
        # Eq. (237): Subgradient of ||Ax - b||_1
        if A is None or b is None:
            return np.zeros_like(x)

        residual = np.dot(A, x) - b
        psi_val = self._psi(residual)
        coeffs = 2.0 * psi_val - 1.0
        
        return np.dot(A.T, coeffs)

    def solve(self, problem, x_init):
        # Eq. (224): Differential inclusion discretization
        x = np.array(x_init, dtype=np.float64)
        history = {'f_x': [], 'x': []}
        start_time = time.time()

        print(f"--- Starting RNN Solver [Max Iter: {self.max_iter}, Step: {self.step_size}] ---")

        for k in range(self.max_iter):
            # 1. Evaluate Problem Functions
            f_val = problem.calc_f(x)
            grad_f = problem.calc_grad_f(x)
            g_val = problem.calc_g(x)
            grad_g = problem.calc_grad_g(x)

            # 2. Compute Neurodynamic Terms
            c_term = self._compute_c(x, g_val, problem.A, problem.b)
            term_obj = -1.0 * c_term * grad_f
            
            grad_P = self._subgradient_P(g_val, grad_g)
            term_ineq = -1.0 * grad_P
            
            grad_Eq = self._subgradient_equality(x, problem.A, problem.b)
            term_eq = -1.0 * grad_Eq

            # 3. Update Direction
            direction = term_obj + term_ineq + term_eq

            # 4. Euler Update
            x_next = x + self.step_size * direction

            # Logging
            history['f_x'].append(f_val)
            history['x'].append(x.copy())

            x = x_next

        elapsed = time.time() - start_time
        return x, history, elapsed

if __name__ == "__main__":
    # --- Configuration Block (Tuned for Convergence) ---
    
    # Change this ID to run different examples (1, 2, 3, 4)
    EXAMPLE_ID = 2
    
    if EXAMPLE_ID == 1:
        # Example 1: Nonconvex 2D
        problem = MathExample1()
        x_init = np.array([0.5, 0.5])
        max_iter = 2000
        step_size = 0.005

    elif EXAMPLE_ID == 2:
        # Example 2: Nonsmooth Pseudoconvex 4D
        # Tuned: Warm start to avoid local minima
        problem = MathExample2()
        x_init = np.array([-1.0, 0.5, -0.5, 0.0])
        max_iter = 10000
        step_size = 0.001

    elif EXAMPLE_ID == 3:
        # Example 3: Convex n-Dim
        dim = 10
        problem = MathExample3(n=dim)
        x_init = np.ones(dim) * 2.0
        max_iter = 1000
        step_size = 0.001

    elif EXAMPLE_ID == 4:
        # Example 4: Pseudoconvex Large Scale
        # Tuned: Smart init (0.8) to satisfy equality constraints immediately
        dim = 10
        problem = MathExample4(n=dim)
        x_init = np.ones(dim) * 0.8
        max_iter = 5000
        step_size = 0.01

    else:
        raise ValueError("Invalid Example ID")

    # --- Execution ---
    rnn = NeurodynamicRNN(step_size=step_size, max_iter=max_iter)
    x_opt, hist, run_time = rnn.solve(problem, x_init)

    print(f"Example {EXAMPLE_ID} Results:")
    print(f"  Optimal X (Head): {x_opt[:5]}")
    print(f"  Optimal f(x): {problem.calc_f(x_opt)}")
    print(f"  Time: {run_time:.4f}s")