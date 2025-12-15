import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import matplotlib.pyplot as plt
from gda import run_gda_solve
from rnn import run_rnn_solver

# =============================================================================
# EXAMPLE 1: 2D Constrained Optimization
# =============================================================================

def obj_func_ex1(x):
    x = x.to(torch.float64)
    x1, x2 = x[0], x[1]
    return (x1**2 + x2**2 + 3) / (1 + 2 * x1 + 8 * x2)

def grad_func_ex1(x):
    xv = x.clone().detach().requires_grad_(True)
    f = obj_func_ex1(xv)
    f.backward()
    return xv.grad.detach()

def constraint_x1_nonneg(x):
    """g1: -x1 <= 0  =>  x1 >= 0"""
    return -x[0]

def constraint_x2_nonneg(x):
    """g2: -x2 <= 0  =>  x2 >= 0"""
    return -x[1]

def constraint_nonlinear_ex1(x):
    """g3: 4 - x1^2 - 2*x1*x2 <= 0  =>  x1^2 + 2*x1*x2 >= 4"""
    x = x.to(torch.float64)
    return 4 - x[0]**2 - 2*x[0]*x[1]

constraints_ex1 = [constraint_x1_nonneg, constraint_x2_nonneg, constraint_nonlinear_ex1]

def proj_func_ex1(x):
    """Projection for Example 1"""
    with torch.no_grad():
        x = x.to(torch.float64)
        x = torch.maximum(x, torch.tensor(0.0, dtype=torch.float64))
        
        for _ in range(20):
            x1, x2 = x[0], x[1]
            constraint_val = x1 ** 2 + 2 * x1 * x2 - 4
            if constraint_val >= -1e-6:
                break
            
            grad_c = torch.tensor([2 * x1 + 2 * x2, 2 * x1], dtype=torch.float64)
            norm_grad_sq = torch.sum(grad_c ** 2)
            if norm_grad_sq < 1e-6:
                break
            
            step = - (constraint_val / norm_grad_sq) * grad_c
            x = x + step
            x = torch.maximum(x, torch.tensor(0.0, dtype=torch.float64))
    return x


def run_comparison():
    print("="*70)
    print("EXAMPLE 1: 2D Constrained Optimization")
    print("="*70)
    
    # Feasible starting point, but far from optimum: x1^2 + 2*x1*x2 = 9 + 12 = 21 >= 4 ✓
    x0_ex1 = torch.tensor([3.0, 2.0], dtype=torch.float64)
    
    print("\n[GDA] Running...")
    x_gda_ex1, hist_gda_ex1 = run_gda_solve(
        grad_func_ex1, proj_func_ex1, obj_func_ex1, x0_ex1,
        step_size=1, sigma=0.1, kappa=0.5, max_iter=2000, tol=1e-8, return_history=True
    )
    print(f"  x_opt = {x_gda_ex1.numpy()}")
    print(f"  f(x_opt) = {obj_func_ex1(x_gda_ex1):.8f}")
    print(f"  Iterations: {len(hist_gda_ex1['iterations'])}")
    print(f"  Time: {hist_gda_ex1['time'][-1]:.4f}s")
    
    print("\n[RNN] Running...")
    x_rnn_ex1, hist_rnn_ex1 = run_rnn_solver(
        grad_func_ex1, constraints_ex1, obj_func_ex1, x0_ex1,
        A=None, b=None, beta=1.0, step_size=0.02, max_iter=2000, tol=1e-8, return_history=True
    )
    print(f"  x_opt = {x_rnn_ex1.numpy()}")
    print(f"  f(x_opt) = {obj_func_ex1(x_rnn_ex1):.8f}")
    print(f"  Iterations: {len(hist_rnn_ex1['iterations'])}")
    print(f"  Time: {hist_rnn_ex1['time'][-1]:.4f}s")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Split RNN data into feasible and infeasible
    feasible_iter = [hist_rnn_ex1['iterations'][i] for i in range(len(hist_rnn_ex1['feasible'])) if hist_rnn_ex1['feasible'][i]]
    feasible_obj = [hist_rnn_ex1['obj'][i] for i in range(len(hist_rnn_ex1['feasible'])) if hist_rnn_ex1['feasible'][i]]
    infeasible_iter = [hist_rnn_ex1['iterations'][i] for i in range(len(hist_rnn_ex1['feasible'])) if not hist_rnn_ex1['feasible'][i]]
    infeasible_obj = [hist_rnn_ex1['obj'][i] for i in range(len(hist_rnn_ex1['feasible'])) if not hist_rnn_ex1['feasible'][i]]
    
    feasible_time = [hist_rnn_ex1['time'][i] for i in range(len(hist_rnn_ex1['feasible'])) if hist_rnn_ex1['feasible'][i]]
    infeasible_time = [hist_rnn_ex1['time'][i] for i in range(len(hist_rnn_ex1['feasible'])) if not hist_rnn_ex1['feasible'][i]]
    
    # By Iterations
    ax = axes[0]
    ax.plot(hist_gda_ex1['iterations'], hist_gda_ex1['obj'], 'b-', linewidth=2, label='GDA')
    if infeasible_iter:
        ax.plot(infeasible_iter, infeasible_obj, '.', color='orange', markersize=3, alpha=0.5, label='RNN (infeasible)')
    if feasible_iter:
        ax.plot(feasible_iter, feasible_obj, 'r-', linewidth=2, label='RNN (feasible)')
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Objective Value f(x)', fontsize=12)
    ax.set_title('Example 1: Convergence by Iterations', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # By Time
    ax = axes[1]
    ax.plot(hist_gda_ex1['time'], hist_gda_ex1['obj'], 'b-', linewidth=2, label='GDA')
    if infeasible_time:
        ax.plot(infeasible_time, infeasible_obj, '.', color='orange', markersize=3, alpha=0.5, label='RNN (infeasible)')
    if feasible_time:
        ax.plot(feasible_time, feasible_obj, 'r-', linewidth=2, label='RNN (feasible)')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Objective Value f(x)', fontsize=12)
    ax.set_title('Example 1: Convergence by Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_example1.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*70)
    print("Plot saved: convergence_example1.png")
    print("="*70)
    plt.show()


if __name__ == "__main__":
    run_comparison()
