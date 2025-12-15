"""
Standard Gradient Descent (GD) Algorithm with Fixed Step Size

Implements:
    x^(k+1) = P_C[x^k - λ * ∇f(x^k)]

where:
    - λ is FIXED (no adaptive adjustment)
    - P_C is projection onto constraint set C
"""

import torch
import time


def run_gd_solve(obj_func, grad_func, proj_func, x0, step_size,
                 max_iter=1000, tol=1e-6, return_history=False):
    """
    Standard Gradient Descent with Fixed Step Size
    
    Parameters:
    -----------
    obj_func : callable
        Objective function f(x)
    grad_func : callable
        Gradient function ∇f(x)
    proj_func : callable
        Projection function P_C(x) onto constraint set C
        If None, no projection is applied (unconstrained)
    x0 : torch.Tensor
        Initial point
    step_size : float
        Fixed step size λ (does NOT change during iterations)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    return_history : bool
        If True, return full iteration history
    
    Returns:
    --------
    x : torch.Tensor
        Final solution
    result : dict or history : dict
        If return_history=False: {'x', 'objective', 'iterations'}
        If return_history=True: (x, {'iterations', 'time', 'obj'})
    """
    
    x = x0.clone().detach()
    start_time = time.time()
    
    # Initialize history tracking
    history = {
        'iterations': [],
        'time': [],
        'obj': []
    } if return_history else None
    
    # Record initial point
    if return_history:
        history['iterations'].append(0)
        history['time'].append(0.0)
        history['obj'].append(obj_func(x).item())
    
    converged = False
    for k in range(max_iter):
        # If already converged but return_history=True, just keep recording same value
        if converged and return_history:
            history['iterations'].append(k + 1)
            history['time'].append(time.time() - start_time)
            history['obj'].append(history['obj'][-1])  # Keep same objective value
            continue
        
        # Compute gradient at current point
        g = grad_func(x)
        
        # Gradient descent step with FIXED step size
        x_new = x - step_size * g
        
        # Project onto constraint set (if projection function provided)
        if proj_func is not None:
            x_new = proj_func(x_new)
        
        # Record history if requested
        if return_history:
            history['iterations'].append(k + 1)
            history['time'].append(time.time() - start_time)
            history['obj'].append(obj_func(x_new).item())
        
        # Check convergence: ||x^(k+1) - x^k|| < tol
        if torch.norm(x_new - x) < tol:
            x = x_new
            if return_history:
                converged = True  # Mark as converged but continue recording
            else:
                break  # Only break if not tracking history
        else:
            x = x_new
    
    if return_history:
        return x, history
    else:
        result = {
            'x': x,
            'objective': obj_func(x).item(),
            'iterations': len(history['iterations']) if history else k + 1
        }
        return x, result


if __name__ == "__main__":
    # Simple test: minimize f(x) = x^2 subject to x >= 0
    print("Testing GD algorithm...")
    print("Problem: min f(x) = x^2 subject to x >= 0")
    print()
    
    def obj_func(x):
        return x ** 2
    
    def grad_func(x):
        return 2 * x
    
    def proj_func(x):
        return torch.maximum(x, torch.tensor(0.0, dtype=torch.float64))
    
    x0 = torch.tensor(5.0, dtype=torch.float64)
    step_size = 0.1
    
    x_opt, result = run_gd_solve(
        obj_func=obj_func,
        grad_func=grad_func,
        proj_func=proj_func,
        x0=x0,
        step_size=step_size,
        max_iter=100,
        tol=1e-6,
        return_history=False
    )
    
    print(f"Initial point: x0 = {x0.item():.6f}")
    print(f"Step size: λ = {step_size}")
    print(f"Optimal solution: x* = {x_opt.item():.6f}")
    print(f"Optimal value: f(x*) = {result['objective']:.6f}")
    print(f"Iterations: {result['iterations']}")
    print()
    print("Expected: x* ≈ 0, f(x*) ≈ 0")
