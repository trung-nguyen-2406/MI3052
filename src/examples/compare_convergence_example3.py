"""
Compare convergence speed between GDA and GD for Example 3
Plot convergence curves for different dimensions n
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import time
sys.path.insert(0, '..')
from gda import run_gda_solve
from gd import run_gd_solve

import warnings

# Tắt các cảnh báo liên quan đến Deprecation (tính năng sắp bỏ)
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(divide='ignore', invalid='ignore')


def create_problem(n):
    """Create problem instance for dimension n"""
    # Parameters
    beta = 0.741271
    alpha = 3 * (beta ** 1.5) * np.sqrt(n + 1)
    
    # Random vectors (fixed seed for reproducibility)
    torch.manual_seed(42)
    a = torch.randn(n)
    e = torch.ones(n)
    
    # Objective function
    def obj_func(x):
        term1 = torch.dot(a, x)
        term2 = alpha * torch.dot(x, x)
        xx = torch.dot(x, x)
        term3 = (beta / torch.sqrt(1 + beta * xx)) * torch.dot(e, x)
        return term1 + term2 + term3
    
    # Gradient function
    def grad_func(x):
        grad1 = a
        grad2 = 2 * alpha * x
        
        xx = torch.dot(x, x)
        denom = torch.sqrt(1 + beta * xx)
        grad3_part1 = (beta / denom) * e
        grad3_part2 = -(beta**2 * torch.dot(e, x) / (2 * denom**3)) * x
        grad3 = grad3_part1 + grad3_part2
        
        return grad1 + grad2 + grad3
    
    # Projection function: x >= 1
    # def proj_func(x):
    #     return torch.clamp(x, min=1.0)

    def proj_func(x):
        from scipy.optimize import root_scalar
        """
        Phép chiếu vector x lên tập C = {z in R++^n : prod(z) >= 1}.
        Bài toán: min ||z - x||^2 s.t. sum(log(z_i)) >= 0
        """
        # 1. Kiểm tra nếu x đã thuộc C (Tích x_i >= 1 hay Tổng log(x_i) >= 0)
        # Sử dụng log để tránh tràn số (overflow) với n lớn
        if torch.sum(torch.log(x)) >= 0:
            return x

        # 2. Nếu không, điểm chiếu nằm trên biên (prod(z) = 1)
        # Chúng ta cần tìm nhân tử Lagrange mu > 0 sao cho:
        # sum( log( (x_i + sqrt(x_i^2 + 4*mu)) / 2 ) ) = 0
        
        def equation(mu):
            # Công thức nghiệm từ điều kiện KKT: z_i = (x_i + sqrt(x_i^2 + 4*mu)) / 2
            # mu là biến số cần tìm
            z = (x + torch.sqrt(x**2 + 4 * mu)) / 2.0
            return torch.sum(torch.log(z))

        # 3. Giải phương trình tìm mu (mu phải dương)
        # Hàm log tăng dần theo mu, nên ta có thể dùng phương pháp Brent hoặc Bisect
        # Bracket [0, 1e5] là khoảng tìm kiếm, có thể cần chỉnh nếu mu quá lớn
        try:
            sol = root_scalar(equation, bracket=[0, 1e6], method='brentq')
            mu_opt = sol.root
        except ValueError:
            # Trường hợp hiếm: nếu không tìm thấy nghiệm trong khoảng, mở rộng khoảng
            sol = root_scalar(equation, bracket=[0, 1e12], method='brentq')
            mu_opt = sol.root

        # 4. Tính vector kết quả z dựa trên mu tối ưu
        z_projected = (x + np.sqrt(x**2 + 4 * mu_opt)) / 2.0
        
        return z_projected    
    
    # Initial point (same for both algorithms)
    torch.manual_seed(123)
    x0 = torch.ones(n) + torch.rand(n) * 0.1
    
    # Calculate Lipschitz constant: L = 4β^(3/2)√n + 3α
    L = 4 * (beta ** 1.5) * np.sqrt(n) + 3 * alpha
    
    return obj_func, grad_func, proj_func, x0, L, alpha, beta


def plot_convergence_comparison(results_gda, results_gd, n_values):
    """Plot convergence comparison for all dimensions"""
    
    num_plots = len(n_values)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.suptitle('Convergence Comparison: GDA vs GD for Different Dimensions', fontsize=16, y=0.995)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, n in enumerate(n_values):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        gda_hist = results_gda[idx]
        gd_hist = results_gd[idx]
        
        # Plot objective value vs iterations
        ax.semilogy(gda_hist['iterations'], gda_hist['obj'], 'b-', linewidth=2, label='GDA')
        ax.semilogy(gd_hist['iterations'], gd_hist['obj'], 'r--', linewidth=2, label='GD')
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Objective Value f(x)', fontsize=11)
        ax.set_title(f'n = {n}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(num_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('convergence_comparison_example3.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'convergence_comparison_example3.png'")
    plt.show()


def plot_convergence_time(results_gda, results_gd, n_values):
    """Plot convergence vs time"""
    
    num_plots = len(n_values)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.suptitle('Convergence vs Time: GDA vs GD', fontsize=16, y=0.995)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, n in enumerate(n_values):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        gda_hist = results_gda[idx]
        gd_hist = results_gd[idx]
        
        # Plot objective value vs time
        ax.semilogy(gda_hist['time'], gda_hist['obj'], 'b-', linewidth=2, label='GDA')
        ax.semilogy(gd_hist['time'], gd_hist['obj'], 'r--', linewidth=2, label='GD')
        
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Objective Value f(x)', fontsize=11)
        ax.set_title(f'n = {n}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(num_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('convergence_time_example3.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'convergence_time_example3.png'")
    plt.show()


def run_comparison():
    """Run comparison between GDA and GD"""
    
    print("\n" + "="*80)
    print("CONVERGENCE COMPARISON: GDA vs GD for Example 3")
    print("="*80)
    
    # Test different dimensions
    n_values = [10, 20, 50, 100, 200, 500]
    
    print(f"""
Problem Definition:
    f(x) = a^T*x + α*x^T*x + β/√(1 + β*x^T*x) * e^T*x
    
    where:
        β = 0.741271
        α = 3β^(3/2)√(n + 1)
        C = {{x ∈ R^n : x_i ≥ 1 for all i}}
    
Lipschitz constant: L = 4β^(3/2)√n + 3α

Step sizes:
    - GDA: λ₀ = 5/L (adaptive with Armijo-type condition)
    - GD:  λ = 1/L (fixed)
    
Convergence criterion: tolerance = 1e-6
    """)
    
    print("=" * 80)
    print("COMPUTATIONAL RESULTS")
    print("=" * 80)
    print(f"{'n':<8}{'Algorithm':<12}{'f(x*)':<15}{'Iterations':<12}{'Time(s)':<12}")
    print("-" * 80)
    
    results_gda = []
    results_gd = []
    
    for n in n_values:
        # Create problem
        obj_func, grad_func, proj_func, x0, L, alpha, beta = create_problem(n)
        
        # ===== Run GDA =====
        lambda_gda = 5.0 / L  # More aggressive initial step size
        
        x_gda, hist_gda = run_gda_solve(
            grad_func=grad_func,
            proj_func=proj_func,
            obj_func=obj_func,
            x0=torch.ones(n) * 360,
            step_size=lambda_gda,
            sigma=0.1,
            kappa=0.1,
            max_iter=30,
            tol=1e-6,
            return_history=True
        )
        
        results_gda.append(hist_gda)
        
        # ===== Run GD =====
        lambda_gd = 1.0 / L  # Standard 1/L step size for GD
        
        x_gd, hist_gd = run_gd_solve(
            obj_func=obj_func,
            grad_func=grad_func,
            proj_func=proj_func,
            x0=torch.ones(n) * 360,
            step_size=lambda_gd,
            max_iter=30,
            tol=1e-6,
            return_history=True
        )
        
        results_gd.append(hist_gd)
        
        # Get actual number of iterations (length of history)
        gda_iters = len(hist_gda['iterations'])
        gd_iters = len(hist_gd['iterations'])
        
        # Print results with actual iteration count
        print(f"{n:<8}{'GDA':<12}{hist_gda['obj'][-1]:<15.4f}{gda_iters:<12}{hist_gda['time'][-1]:<12.4f}")
        print(f"{' ':<8}{'GD':<12}{hist_gd['obj'][-1]:<15.4f}{gd_iters:<12}{hist_gd['time'][-1]:<12.4f}")
        print()
    
    print("=" * 80)
    
    # Plot comparisons
    print("\nGenerating convergence plots...")
    plot_convergence_comparison(results_gda, results_gd, n_values)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY: Iterations to Convergence")
    print("=" * 80)
    print(f"{'n':<10}{'GDA Iters':<15}{'GD Iters':<15}{'Speedup':<15}")
    print("-" * 80)
    
    for idx, n in enumerate(n_values):
        gda_iters = len(results_gda[idx]['iterations'])
        gd_iters = len(results_gd[idx]['iterations'])
        speedup = gd_iters / gda_iters if gda_iters > 0 else 0
        print(f"{n:<10}{gda_iters:<15}{gd_iters:<15}{speedup:<15.2f}x")
    
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()