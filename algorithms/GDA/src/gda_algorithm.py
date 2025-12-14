"""
Gradient Descent Adaptive (GDA) Algorithm Implementation
Based on: Self-adaptive algorithms for quasiconvex programming
Algorithm 1 from the paper
"""

import numpy as np
from typing import Callable, Optional, Dict, Any
import time


class GDAOptimizer:
    """
    Gradient Descent Adaptive (GDA) Algorithm for nonconvex and quasiconvex optimization
    
    Algorithm 1 (GDA):
    Step 0: Choose x^0 ∈ C, λ_0 ∈ (0, +∞), σ, κ ∈ (0, 1). Set k = 0.
    Step 1: Given x^k and λ_k, compute x^{k+1} and λ_{k+1} as:
            x^{k+1} = P_C(x^k - λ_k ∇f(x^k))
            If f(x^{k+1}) ≤ f(x^k) - σ⟨∇f(x^k), x^k - x^{k+1}⟩ then set λ_{k+1} = λ_k else set λ_{k+1} = κλ_k
    Step 2: Update k := k + 1. If x^{k+1} = x^k then STOP else go to Step 1.
    """
    
    def __init__(self, 
                 func: Callable,
                 grad_func: Callable,
                 x0: np.ndarray,
                 projection_func: Optional[Callable] = None,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 lambda_0: float = 1.0,
                 sigma: float = 0.1,
                 kappa: float = 0.5,
                 verbose: bool = False):
        """
        Initialize GDA optimizer
        
        Parameters:
        -----------
        func : Callable
            Objective function f(x)
        grad_func : Callable
            Gradient function ∇f(x)
        x0 : np.ndarray
            Initial point in feasible set C
        projection_func : Callable, optional
            Projection operator P_C(x). If None, identity projection (unconstrained)
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for convergence
        lambda_0 : float
            Initial step size λ_0
        sigma : float
            Sufficient decrease parameter σ ∈ (0, 1)
        kappa : float
            Step size reduction factor κ ∈ (0, 1)
        verbose : bool
            Print progress information
        """
        self.func = func
        self.grad_func = grad_func
        self.x0 = np.copy(x0)
        self.projection_func = projection_func if projection_func is not None else lambda x: x
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_0 = lambda_0
        self.sigma = sigma
        self.kappa = kappa
        self.verbose = verbose
        
        self.history = {
            'x': [],
            'f': [],
            'grad_norm': [],
            'step_size': [],
            'iterations': 0
        }
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run GDA optimization algorithm
        
        Returns:
        --------
        Dict with optimization results
        """
        x = np.copy(self.x0)
        lambda_k = self.lambda_0
        
        start_time = time.time()
        
        for k in range(self.max_iter):
            # Compute gradient at current point
            grad = self.grad_func(x)
            grad_norm = np.linalg.norm(grad)
            
            # Current function value
            f_val = self.func(x)
            
            # Store history
            self.history['x'].append(np.copy(x))
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            self.history['step_size'].append(lambda_k)
            
            if self.verbose and (k % 10 == 0 or k < 10):
                print(f"Iter {k}: f(x) = {f_val:.6f}, ||grad|| = {grad_norm:.6e}, lambda = {lambda_k:.6e}")
            
            # Check convergence: if gradient is small enough
            # if grad_norm < self.tol:
            #     if self.verbose:
            #         print(f"Converged at iteration {k}")
            #     break
            
            # Step 1: Compute x^{k+1} = P_C(x^k - λ_k ∇f(x^k))
            x_new = self.projection_func(x - lambda_k * grad)
            f_new = self.func(x_new)
            
            # Compute inner product ⟨∇f(x^k), x^k - x^{k+1}⟩
            direction = x - x_new
            inner_product = np.dot(grad, direction)
            
            # Check sufficient decrease condition:
            # f(x^{k+1}) ≤ f(x^k) - σ⟨∇f(x^k), x^k - x^{k+1}⟩
            if f_new <= f_val - self.sigma * inner_product:
                # Condition satisfied: keep step size
                lambda_k = lambda_k
                # if self.verbose and k < 20:
                #     print(f"  -> Decrease satisfied, keep λ = {lambda_k:.6e}")
            else:
                # Condition not satisfied: reduce step size
                lambda_k = self.kappa * lambda_k
                # if self.verbose and k < 20:
                #     print(f"  -> Decrease failed, reduce λ = {lambda_k:.6e}")
            
            # Update x
            
            # Step 2: Check if converged (x^{k+1} = x^k)
            if np.allclose(x, x_new, atol=self.tol, rtol=1e-5):
                if self.verbose:
                    print(f"Stationary point reached at iteration {k+1}")
                break
            x = x_new
        
        elapsed_time = time.time() - start_time
        self.history['iterations'] = k + 1
        
        result = {
            'x': x,
            'f': self.func(x),
            'iterations': k + 1,
            'time': elapsed_time,
            'history': self.history,
            'converged': grad_norm < self.tol
        }
        
        return result
