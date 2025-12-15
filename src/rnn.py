import torch

def check_feasibility(x, constraints, tol=1e-4):
    """
    Check if x satisfies all constraints g_i(x) <= 0
    
    Returns:
        (is_feasible, max_violation)
    """
    max_violation = 0.0
    for g_func in constraints:
        x_eval = x.clone().detach()
        g_val = float(g_func(x_eval))
        violation = max(0, g_val)
        max_violation = max(max_violation, violation)
    
    is_feasible = max_violation <= tol
    return is_feasible, max_violation

def compute_subgradient_penalty(x, constraints):
    """
    Compute subgradient of P(x) = sum_i max(0, g_i(x))
    
    Args:
        x: current point
        constraints: list of constraint functions g_i(x)
    
    Returns:
        subgradient of P(x)
    """
    x = x.clone().detach().requires_grad_(True)
    subgrad = torch.zeros_like(x)
    
    for g_func in constraints:
        g_val = g_func(x)
        if g_val > 1e-10:  # Active constraint
            # Compute gradient of g_i
            g_val.backward(retain_graph=True)
            subgrad = subgrad + x.grad.clone()
            x.grad.zero_()
    
    return subgrad.detach()


def compute_adjusted_term(x, constraints, epsilon=1e-8):
    """
    Compute adjusted term c(x(t)) = product of c_i(t)
    where c_i(t) in 1 - Psi(J_i(x(t)))
    
    Psi(s) = {1 if s>0, [0,1] if s=0, 0 if s<0}
    J_i(x(t)) includes constraint values
    
    Smooth adjustment based on constraint violations
    """
    c = 1.0
    
    for g_func in constraints:
        x_eval = x.clone().detach()
        g_val = float(g_func(x_eval))
        
        if g_val > epsilon:  # Violated constraint
            # Smooth reduction: c *= exp(-violation)
            violation_factor = min(g_val, 5.0)  # Cap to avoid too small c
            c *= max(0.5, 1.0 - 0.1 * violation_factor)  # Gradual reduction
        elif abs(g_val) <= epsilon:  # Near boundary
            c *= 0.95  # Very gentle reduction
    
    return max(c, 0.5)  # Higher minimum for stability


def compute_equality_subgradient(x, A, b):
    """
    Compute subgradient of ||Ax - b||_1
    
    Returns:
        A^T * sign(Ax - b)
    """
    if A is None or b is None:
        return torch.zeros_like(x)
    
    residual = A @ x - b
    sign_residual = torch.sign(residual)
    return A.T @ sign_residual


def run_rnn_solver(grad_func, constraints, obj_func, x0, 
                   A=None, b=None, beta=1.0,
                   step_size=0.01, max_iter=2000, tol=1e-8, return_history=False):
    """
    RNN-based neurodynamic algorithm for constrained optimization
    
    Solves: min f(x) s.t. g_i(x) <= 0, Ax = b
    
    ODE: dx/dt = -c(x)*grad_f(x) - subgrad_P(x) - beta*subgrad_||Ax-b||_1
    
    Args:
        grad_func: gradient of objective function
        constraints: list of inequality constraint functions g_i(x) <= 0
        obj_func: objective function for monitoring
        x0: initial point
        A, b: equality constraint Ax = b (optional)
        beta: penalty parameter for equality constraint
        step_size: time step h for Euler discretization
        max_iter: maximum iterations
        tol: convergence tolerance
        return_history: if True, returns (x, history) else just x
    
    Returns:
        x_opt: optimal solution (or (x_opt, history) if return_history=True)
    """
    import time
    
    x = x0.clone().detach().to(torch.float64)
    h = step_size
    h_init = step_size  # Store initial step size
    
    if return_history:
        history = {'iterations': [], 'time': [], 'obj': [], 'feasible': []}
        start_time = time.time()
    
    for k in range(max_iter):
        # 1. Compute gradient of objective
        grad_f = grad_func(x)
        
        # 2. Compute adjusted term c(x)
        c_x = compute_adjusted_term(x, constraints)
        
        # 3. Compute subgradient of penalty function P(x)
        subgrad_P = compute_subgradient_penalty(x, constraints)
        
        # 4. Compute subgradient of equality constraint
        subgrad_eq = compute_equality_subgradient(x, A, b)
        
        # 5. RNN update: x_new = x - h * [c(x)*grad_f + subgrad_P + beta*subgrad_eq]
        dx = c_x * grad_f + subgrad_P + beta * subgrad_eq
        x_new = x - h * dx
        
        if return_history:
            # Check feasibility and record everything
            is_feasible, _ = check_feasibility(x_new, constraints, tol=0.015)
            obj_val = float(obj_func(x_new))
            
            history['iterations'].append(k)
            history['time'].append(time.time() - start_time)
            history['obj'].append(obj_val)
            history['feasible'].append(is_feasible)
        
        # 6. Check convergence
        if torch.norm(x_new - x) < tol:
            if return_history:
                return x_new, history
            return x_new
        
        x = x_new
        
        # Simple gradual decay for stability
        if k > 0 and k % 100 == 0:
            h = h * 0.97  # Very gradual decay
    
    if return_history:
        return x, history
    return x



