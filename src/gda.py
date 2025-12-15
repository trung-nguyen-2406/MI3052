import torch
import time

def run_gda_solve(grad_func, proj_func, obj_func, x0, step_size, sigma, kappa, max_iter = 1000, tol = 1e-6, return_history=False):
    """
    GDA Algorithm with Adaptive Step Size
    
    Parameters:
    -----------
    grad_func : callable
        Gradient function
    proj_func : callable
        Projection function onto constraint set (if None, no projection)
    obj_func : callable
        Objective function
    x0 : torch.Tensor
        Initial point
    step_size : float
        Initial step size (lambda_0)
    sigma : float
        Parameter for Armijo condition
    kappa : float
        Step size reduction factor
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    return_history : bool
        If True, returns (x, history) else just x
    """
    x = x0.clone().detach()
    
    if return_history:
        history = {'iterations': [], 'time': [], 'obj': []}
        start_time = time.time()
        # Record initial point
        history['iterations'].append(0)
        history['time'].append(0.0)
        history['obj'].append(float(obj_func(x)))
    
    converged = False
    for k in range(int(max_iter)):
        # If already converged but return_history=True, just keep recording same value
        if converged and return_history:
            history['iterations'].append(k + 1)
            history['time'].append(time.time() - start_time)
            history['obj'].append(history['obj'][-1])  # Keep same objective value
            continue
        
        g = grad_func(x)
        if g is None:
            break
        
        # Gradient step
        y = x - step_size * g
        
        # Project onto constraint set (if projection function provided)
        if proj_func is not None:
            x_new = proj_func(y)
        else:
            x_new = y
        
        # Armijo-type step size adjustment
        if obj_func(x_new) > obj_func(x) - sigma * torch.inner(g, x - x_new):
            step_size = kappa * step_size
        
        if return_history:
            history['iterations'].append(k + 1)
            history['time'].append(time.time() - start_time)
            history['obj'].append(float(obj_func(x_new)))
        
        # Check convergence
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
    return x

# def run_gda_minimax(x0, y0, grad_x, grad_y, proj_x=None, proj_y=None,
#                     step_x=1e-3, step_y=1e-3, obj_func=None,
#                     max_iter=1000, tol=1e-6):
#     """
#     Simultaneous gradient descent in x (min) and ascent in y (max).
#     grad_x(x,y), grad_y(x,y).
#     proj_x/proj_y optional projections.
#     step_x/step_y can be floats or callables(step_iter)->float.
#     obj_func(x,y) optional for history.
#     Returns (x,y,history).
#     """
#     x = x0.clone().detach().float()
#     y = y0.clone().detach().float()
#     history = {'x': [], 'y': [], 'obj': []}
#     for k in range(int(max_iter)):
#         gx = grad_x(x, y)
#         gy = grad_y(x, y)
#         ax = step_x(k) if callable(step_x) else step_x
#         ay = step_y(k) if callable(step_y) else step_y
#         x_new = x - ax * gx
#         y_new = y + ay * gy
#         if proj_x is not None:
#             x_new = proj_x(x_new)
#         if proj_y is not None:
#             y_new = proj_y(y_new)
#         history['x'].append(x_new.clone())
#         history['y'].append(y_new.clone())
#         if obj_func is not None:
#             history['obj'].append(float(obj_func(x_new, y_new)))
#         # stopping
#         if torch.norm(x_new - x) < tol and torch.norm(y_new - y) < tol:
#             x, y = x_new, y_new
#             break
#         x, y = x_new, y_new
#     return x, y, history
# # ...existing code...