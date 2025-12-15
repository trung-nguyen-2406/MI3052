import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import matplotlib.pyplot as plt
from gda import run_gda_solve
from rnn import run_rnn_solver

# =============================================================================
# EXAMPLE 2: 4D Constrained Optimization
# =============================================================================

def theta_smooth(x, mu=0.1):
    """Smooth approximation of |x|"""
    if torch.abs(x) >= (mu / 2):
        return torch.abs(x)
    else:
        return x**2 / mu + mu / 4

def obj_func_ex2(x, mu=0.01):
    """Smoothed objective function"""
    x = x.to(torch.float64)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return (torch.exp(theta_smooth(x2 - 3, mu)) - 30) / (x1 ** 2 + x3 ** 2 + 2 * x4 ** 2 + 4)

def grad_func_ex2(x):
    xv = x.clone().detach().requires_grad_(True)
    f = obj_func_ex2(xv, mu=0.01)
    f.backward()
    return xv.grad.detach()

def constraint_g1_ex2(x):
    """g1(x) = (x1 + x3)^3 + 2*x2^2 - 10 <= 0"""
    x1, x2, x3 = x[0], x[1], x[2]
    return (x1 + x3)**3 + 2 * x2**2 - 10.0

def constraint_g2_ex2(x):
    """g2(x) = (x2 - 1)^2 - 1 <= 0"""
    x2 = x[1]
    return (x2 - 1)**2 - 1.0

def constraint_h_ex2(x):
    """h(x) = 2x1 + 4x2 + x3 + 1 = 0 (equality, convert to two inequalities)"""
    x1, x2, x3 = x[0], x[1], x[2]
    h_val = 2 * x1 + 4 * x2 + x3 + 1.0
    return h_val

def constraint_h_ex2_pos(x):
    """h(x) <= 0"""
    return constraint_h_ex2(x)

def constraint_h_ex2_neg(x):
    """h(x) >= 0  =>  -h(x) <= 0"""
    return -constraint_h_ex2(x)

constraints_ex2 = [constraint_g1_ex2, constraint_g2_ex2, constraint_h_ex2_pos, constraint_h_ex2_neg]

def proj_func_ex2(y_point, max_iters=100, learning_rate=0.01):
    """Improved projection using penalty method with better convergence"""
    device = torch.device("cpu")
    x = y_point.clone().detach().to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([x], lr=learning_rate, max_iter=20, line_search_fn='strong_wolfe')
    y_target = y_point.detach()

    def closure():
        optimizer.zero_grad()
        objective_loss = 0.5 * torch.sum((x - y_target)**2)
        
        h_x = constraint_h_ex2(x)
        equality_penalty = 10000 * (h_x**2)
        
        g1_x = constraint_g1_ex2(x)
        g2_x = constraint_g2_ex2(x)
        inequality_penalty = 10000 * (torch.max(torch.tensor(0.0).to(device), g1_x)**2 + 
                                      torch.max(torch.tensor(0.0).to(device), g2_x)**2)
        
        total_loss = objective_loss + equality_penalty + inequality_penalty
        total_loss.backward()
        return total_loss
    
    optimizer.step(closure)
    return x.detach()


def run_comparison():
    print("="*70)
    print("EXAMPLE 2: 4D Constrained Optimization")
    print("="*70)
    
    x0_ex2 = torch.tensor([0.5, 1.5, -0.5, 0.8], dtype=torch.float64)
    
    print("\n[GDA] Running...")
    x_gda_ex2, hist_gda_ex2 = run_gda_solve(
        grad_func_ex2, proj_func_ex2, lambda x: obj_func_ex2(x, mu=0.01), x0_ex2,
        step_size=0.1, sigma=0.1, kappa=0.5, max_iter=2000, tol=1e-6, return_history=True
    )
    print(f"  x_opt = {x_gda_ex2.numpy()}")
    print(f"  f(x_opt) = {obj_func_ex2(x_gda_ex2, mu=0.01):.8f}")
    print(f"  Iterations: {len(hist_gda_ex2['iterations'])}")
    print(f"  Time: {hist_gda_ex2['time'][-1]:.4f}s")
    
    print("\n[RNN] Running...")
    x_rnn_ex2, hist_rnn_ex2 = run_rnn_solver(
        grad_func_ex2, constraints_ex2, lambda x: obj_func_ex2(x, mu=0.01), x0_ex2,
        A=None, b=None, beta=5.0, step_size=0.01, max_iter=3000, tol=1e-6, return_history=True
    )
    print(f"  x_opt = {x_rnn_ex2.numpy()}")
    print(f"  f(x_opt) = {obj_func_ex2(x_rnn_ex2, mu=0.01):.8f}")
    print(f"  Iterations: {len(hist_rnn_ex2['iterations'])}")
    print(f"  Time: {hist_rnn_ex2['time'][-1]:.4f}s")
    
    # Check feasibility statistics
    feasible_count = sum(1 for f in hist_rnn_ex2['feasible'] if f)
    print(f"  Feasible iterations: {feasible_count}/{len(hist_rnn_ex2['feasible'])}")
    
    # Check final constraint violations
    print(f"  Final constraint violations:")
    for i, g in enumerate(constraints_ex2):
        g_val = float(g(x_rnn_ex2))
        print(f"    g{i+1}(x) = {g_val:.6e} (should be <= 0)")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Split RNN data into feasible and infeasible
    feasible_iter = [hist_rnn_ex2['iterations'][i] for i in range(len(hist_rnn_ex2['feasible'])) if hist_rnn_ex2['feasible'][i]]
    feasible_obj = [hist_rnn_ex2['obj'][i] for i in range(len(hist_rnn_ex2['feasible'])) if hist_rnn_ex2['feasible'][i]]
    infeasible_iter = [hist_rnn_ex2['iterations'][i] for i in range(len(hist_rnn_ex2['feasible'])) if not hist_rnn_ex2['feasible'][i]]
    infeasible_obj = [hist_rnn_ex2['obj'][i] for i in range(len(hist_rnn_ex2['feasible'])) if not hist_rnn_ex2['feasible'][i]]
    
    feasible_time = [hist_rnn_ex2['time'][i] for i in range(len(hist_rnn_ex2['feasible'])) if hist_rnn_ex2['feasible'][i]]
    infeasible_time = [hist_rnn_ex2['time'][i] for i in range(len(hist_rnn_ex2['feasible'])) if not hist_rnn_ex2['feasible'][i]]
    
    # By Iterations
    ax = axes[0]
    ax.plot(hist_gda_ex2['iterations'], hist_gda_ex2['obj'], 'b-', linewidth=2, label='GDA')
    if infeasible_iter:
        ax.plot(infeasible_iter, infeasible_obj, '.', color='orange', markersize=3, alpha=0.5, label='RNN (infeasible)')
    if feasible_iter:
        ax.plot(feasible_iter, feasible_obj, 'r-', linewidth=2, label='RNN (feasible)')
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Objective Value f(x)', fontsize=12)
    ax.set_title('Example 2: Convergence by Iterations', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # By Time
    ax = axes[1]
    ax.plot(hist_gda_ex2['time'], hist_gda_ex2['obj'], 'b-', linewidth=2, label='GDA')
    if infeasible_time:
        ax.plot(infeasible_time, infeasible_obj, '.', color='orange', markersize=3, alpha=0.5, label='RNN (infeasible)')
    if feasible_time:
        ax.plot(feasible_time, feasible_obj, 'r-', linewidth=2, label='RNN (feasible)')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Objective Value f(x)', fontsize=12)
    ax.set_title('Example 2: Convergence by Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_example2.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*70)
    print("Plot saved: convergence_example2.png")
    print("="*70)
    plt.show()


if __name__ == "__main__":
    run_comparison()
