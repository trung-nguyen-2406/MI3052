import torch

def run_gda_solve(grad_func, proj_func, obj_func, x0, step_size, sigma, kappa, max_iter = 1000, tol = 1e-6):
    """
    grad_func(x) -> gradient tensor
    proj_func(x) -> projected x (or None for identity)
    obj_func(x) -> scalar objective (for history)
    step_size (lamda0) can be float  -> float
    Returns x where history is dict with lists 'x' and 'obj'.
    """
    x = x0.clone().detach()
    # history = {'x': [], 'obj': []}
    for k in range(int(max_iter)):
        g = grad_func(x)
        if g is None:
            break
        x_new = proj_func(x - step_size * g)
        step_size = kappa * step_size if (obj_func(x_new) > obj_func(x) - sigma * torch.inner(g, x - x_new)) else step_size

        if (torch.norm(x_new - x) < tol) : return x_new
        x = x_new
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