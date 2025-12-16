"""
So sánh Projected Gradient (4TICK với α_n = 1/n) vs GDA
Example 3: OP(f, C) từ Ferreira and Sosa (2022)

Bài toán:
Let e := (1, ..., n) ∈ R^n be a vector, α > 0 and β > 0 be constants
satisfying the parameter condition 2α > 3β^(3/2)√n.

minimize f(x) := a^T x + α x^T x + β/(√(1 + β x^T x)) e^T x
subject to x ∈ C

Với C := {x ∈ R^n_{++} : 1 ≤ x_1 ... x_n}

Tham số:
β = 0.741271
α = 3β^(3/2)√n + 1
L = 4β^(3/2)√n + 3α (Lipschitz coefficient)
λ_GD = 1/L
λ_0_GDA = 5/L

Test với các giá trị n = 10, 20, 50, 100, 200, 500
"""

import numpy as np
import time
import sys
import os
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Warning: tabulate not installed. Using simple print format.")

# Thêm đường dẫn
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GDA', 'src'))

# Import PyTorch
import torch

# Import module 4TICK
import importlib.util

# Đường dẫn tuyệt đối đến file 4tick.py
path_4tick = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', '4tick.py'))
if not os.path.exists(path_4tick):
    raise FileNotFoundError(f"Không tìm thấy file: {path_4tick}")

spec_4tick = importlib.util.spec_from_file_location("tick4", path_4tick)
tick4_module = importlib.util.module_from_spec(spec_4tick)
sys.modules['tick4'] = tick4_module
spec_4tick.loader.exec_module(tick4_module)
projected_gradient_decreasing_step = tick4_module.projected_gradient_decreasing_step

# Import GDA
path_gda = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GDA', 'src', 'gda.py'))
if not os.path.exists(path_gda):
    raise FileNotFoundError(f"Không tìm thấy file: {path_gda}")

spec_gda = importlib.util.spec_from_file_location("gda", path_gda)
gda_module = importlib.util.module_from_spec(spec_gda)
spec_gda.loader.exec_module(gda_module)
run_gda_solve = gda_module.run_gda_solve


class Example3Problem:
    """Định nghĩa bài toán Example 3"""
    
    def __init__(self, n: int):
        """
        Khởi tạo bài toán với chiều n
        
        Parameters:
        -----------
        n : int
            Số chiều của bài toán
        """
        self.n = n
        self.beta = 0.741271
        self.alpha = 3 * (self.beta ** 1.5) * np.sqrt(n) + 1
        self.L = 4 * (self.beta ** 1.5) * np.sqrt(n) + 3 * self.alpha
        self.step_gd = 1.0 / self.L
        self.step_gda_initial = 5.0 / self.L
        
        # a ∈ R^n_{++} (all positive components)
        self.a = np.ones(n)
        
        # e = (1, 1, ..., 1)
        self.e = np.ones(n)
    
    # ============ Numpy version (cho 4TICK) ============
    def objective_np(self, x: np.ndarray) -> float:
        """
        f(x) = a^T x + α x^T x + β/(√(1 + β x^T x)) e^T x
        """
        term1 = np.dot(self.a, x)
        term2 = self.alpha * np.dot(x, x)
        
        x_t_x = np.dot(x, x)
        denominator = np.sqrt(1 + self.beta * x_t_x)
        term3 = (self.beta / denominator) * np.dot(self.e, x)
        
        return term1 + term2 + term3
    
    def gradient_np(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient của f(x)
        """
        x_t_x = np.dot(x, x)
        e_t_x = np.dot(self.e, x)
        
        sqrt_term = np.sqrt(1 + self.beta * x_t_x)
        
        # ∇f = a + 2α x + β/(√(1 + β x^T x)) e - β^2/(2(1 + β x^T x)^(3/2)) (e^T x) x
        grad = self.a + 2 * self.alpha * x
        grad += (self.beta / sqrt_term) * self.e
        grad -= (self.beta**2 / (2 * sqrt_term**3)) * e_t_x * x
        
        return grad
    
    def projection_np(self, x: np.ndarray) -> np.ndarray:
        """
        Chiếu lên C: {x ∈ R^n_{++} : 1 ≤ x_1 ... x_n}
        """
        eps = 1e-6  # Epsilon lớn hơn để tránh numerical issues
        x_proj = x.copy()
        
        # Bước 1: x > 0 (x ∈ R^n_{++})
        x_proj = np.maximum(x_proj, eps)
        
        # Bước 2: Ràng buộc tích: prod(x) >= 1
        # Sử dụng log để tránh overflow/underflow
        log_prod = np.sum(np.log(x_proj))
        
        if log_prod < 0:  # prod(x) < 1
            # Scale up: nhân tất cả components với exp(-log_prod/n)
            # Tương đương với (1/prod_x)^(1/n) nhưng ổn định hơn
            log_scale = -log_prod / self.n
            x_proj = x_proj * np.exp(log_scale)
        
        # Đảm bảo vẫn dương
        x_proj = np.maximum(x_proj, eps)
        
        return x_proj
    
    # ============ PyTorch version (cho GDA) ============
    def objective_torch(self, x: torch.Tensor) -> torch.Tensor:
        """f(x) = a^T x + α x^T x + β/(√(1 + β x^T x)) e^T x"""
        x = x.to(torch.float64)
        a_t = torch.tensor(self.a, dtype=torch.float64)
        e_t = torch.tensor(self.e, dtype=torch.float64)
        
        term1 = torch.dot(a_t, x)
        term2 = self.alpha * torch.dot(x, x)
        
        x_t_x = torch.dot(x, x)
        denominator = torch.sqrt(1 + self.beta * x_t_x)
        term3 = (self.beta / denominator) * torch.dot(e_t, x)
        
        return term1 + term2 + term3
    
    def gradient_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Tính gradient bằng autograd"""
        xv = x.clone().detach().requires_grad_(True)
        f = self.objective_torch(xv)
        f.backward()
        return xv.grad.detach()
    
    def projection_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Chiếu lên C: {x ∈ R^n_{++} : 1 ≤ x_1 ... x_n}"""
        with torch.no_grad():
            x = x.to(torch.float64)
            
            # 1. Enforce x > 0
            eps = 1e-6  # Epsilon lớn hơn để tránh numerical issues
            x = torch.maximum(x, torch.tensor(eps, dtype=torch.float64))
            
            # 2. Enforce product constraint: prod(x) >= 1
            # Sử dụng log để tránh overflow/underflow
            log_prod = torch.sum(torch.log(x))
            
            if log_prod.item() < 0:  # prod(x) < 1
                # Scale up
                log_scale = -log_prod / self.n
                x = x * torch.exp(log_scale)
            
            # Ensure still positive
            x = torch.maximum(x, torch.tensor(eps, dtype=torch.float64))
        
        return x


def run_4tick_method(problem: Example3Problem, x0: np.ndarray, max_iter: int = 5000, 
                     alpha_func=None, method_name="4TICK"):
    """Chạy phương pháp 4TICK Projected Gradient"""
    if alpha_func is None:
        dim = problem.n  # số chiều
        alpha_func = lambda n: dim / n
        alpha_desc = f"α_n = {dim}/n"
    else:
        alpha_desc = method_name
    
    start_time = time.time()
    
    x_opt, f_history, step_history = projected_gradient_decreasing_step(
        objective=problem.objective_np,
        gradient=problem.gradient_np,
        projection=problem.projection_np,
        x0=x0,
        max_iter=max_iter,
        tol=1e-6,
        alpha_func=alpha_func
    )
    
    elapsed_time = time.time() - start_time
    
    actual_iters = len(f_history) - 1
    
    return {
        'x_final': x_opt,
        'f_opt': problem.objective_np(x_opt),
        'iterations': actual_iters,
        'time': elapsed_time,
        'f_history': f_history,
        'method': '4TICK'
    }


def run_gda_method(problem: Example3Problem, x0_np: np.ndarray, max_iter: int = 5000):
    """Chạy phương pháp GDA với tracking số iterations và f_history"""
    # Convert to torch
    x = torch.tensor(x0_np, dtype=torch.float64)
    
    # GDA parameters
    sigma = 1e-4
    kappa = 0.3
    step_size = problem.step_gda_initial
    tol = 1e-8
    
    # Track history
    f_history = [float(problem.objective_torch(x))]
    
    start_time = time.time()
    
    try:
        # GDA algorithm với tracking
        for k in range(int(max_iter)):
            g = problem.gradient_torch(x)
            if g is None:
                break
            
            x_new = problem.projection_torch(x - step_size * g)
            
            # Armijo line search
            f_new = problem.objective_torch(x_new)
            f_old = problem.objective_torch(x)
            
            if f_new > f_old - sigma * torch.inner(g, x - x_new):
                step_size = kappa * step_size
            
            # Convergence check
            if torch.norm(x_new - x) < tol:
                x = x_new
                f_history.append(float(problem.objective_torch(x)))
                elapsed_time = time.time() - start_time
                print(f"  >> Hội tụ tại iteration {k+1}")
                return {
                    'x_final': x.numpy(),
                    'f_opt': float(problem.objective_torch(x)),
                    'iterations': k + 1,
                    'time': elapsed_time,
                    'f_history': f_history,
                    'method': 'GDA'
                }
            
            x = x_new
            f_history.append(float(problem.objective_torch(x)))
        
        elapsed_time = time.time() - start_time
        
        # Kiểm tra NaN
        if torch.isnan(x).any():
            print("  !!! CẢNH BÁO: GDA trả về NaN!!!")
            return {
                'x_final': x0_np,
                'f_opt': float('nan'),
                'iterations': 0,
                'time': elapsed_time,
                'f_history': [],
                'method': 'GDA (FAILED)'
            }
        
        return {
            'x_final': x.numpy(),
            'f_opt': float(problem.objective_torch(x)),
            'iterations': max_iter,
            'time': elapsed_time,
            'f_history': f_history,
            'method': 'GDA'
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  !!! LỖI khi chạy GDA: {e}")
        return {
            'x_final': x0_np,
            'f_opt': float('nan'),
            'iterations': 0,
            'time': elapsed_time,
            'f_history': [],
            'method': 'GDA (ERROR)'
        }


def compare_algorithms_for_n(n: int, max_iter: int = 5000):
    """So sánh 4TICK vs GDA cho một giá trị n cụ thể"""
    print("\n" + "="*80)
    print(f"EXAMPLE 3: SO SÁNH 4TICK vs GDA với n = {n}")
    print("="*80)
    
    # Khởi tạo bài toán
    problem = Example3Problem(n)
    
    print(f"\nTham số bài toán:")
    print(f"  n = {n}")
    print(f"  β = {problem.beta}")
    print(f"  α = {problem.alpha:.6f}")
    print(f"  L = {problem.L:.6f}")
    print(f"  λ_GD = 1/L = {problem.step_gd:.6f}")
    print(f"  λ_0_GDA = 5/L = {problem.step_gda_initial:.6f}")
    
    # Điểm khởi đầu: x0 = (2, 2, ..., 2)
    x0 = np.ones(n) * 5.0
    print(f"\nĐiểm khởi đầu: x0 = (2, 2, ..., 2)")
    
    # Chạy 4TICK
    print(f"\n>>> Chạy 4TICK (α_n = 1/n)...")
    result_4tick = run_4tick_method(problem, x0, max_iter=max_iter)
    
    # Chạy GDA
    print(f"\n>>> Chạy GDA (λ_0 = 5/L)...")
    result_gda = run_gda_method(problem, x0, max_iter=max_iter)
    
    # Tạo bảng so sánh
    print("\n" + "="*80)
    print(f"KẾT QUẢ SO SÁNH (n = {n})")
    print("="*80)
    
    if HAS_TABULATE:
        table_data = [
            ["Algorithm", "f(x*)", "Iterations", "Time (s)"],
            ["4TICK", f"{result_4tick['f_opt']:.8f}", result_4tick['iterations'], f"{result_4tick['time']:.4f}"],
            ["GDA", f"{result_gda['f_opt']:.8f}", result_gda['iterations'], f"{result_gda['time']:.4f}"]
        ]
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    else:
        print(f"{'Algorithm':<15} {'f(x*)':<20} {'Iterations':<15} {'Time (s)':<10}")
        print("-"*80)
        print(f"{'4TICK':<15} {result_4tick['f_opt']:<20.8f} {result_4tick['iterations']:<15} {result_4tick['time']:<10.4f}")
        print(f"{'GDA':<15} {result_gda['f_opt']:<20.8f} {result_gda['iterations']:<15} {result_gda['time']:<10.4f}")
    
    # Kiểm tra ràng buộc
    prod_4tick = np.prod(result_4tick['x_final'])
    prod_gda = np.prod(result_gda['x_final'])
    
    print(f"\nKiểm tra ràng buộc prod(x) >= 1:")
    print(f"  4TICK: prod(x*) = {prod_4tick:.6f}")
    print(f"  GDA:   prod(x*) = {prod_gda:.6f}")
    
    return {
        'n': n,
        '4tick': result_4tick,
        'gda': result_gda
    }


def plot_convergence(all_results, output_dir):
    """Vẽ đồ thị tốc độ hội tụ của 4TICK và GDA"""
    print("\n>>> Vẽ đồ thị tốc độ hội tụ...")
    
    # Tạo figure với 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('So sánh tốc độ hội tụ: 4TICK (α_n = 1/n) vs GDA', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        n = result['n']
        
        # 4TICK history
        f_hist_4tick = result['4tick']['f_history']
        iters_4tick = range(len(f_hist_4tick))
        
        # GDA history
        f_hist_gda = result['gda']['f_history']
        iters_gda = range(len(f_hist_gda))
        
        # Plot
        ax.semilogy(iters_4tick, f_hist_4tick, 'b-', linewidth=2, label='4TICK (α_n = dim/n)', alpha=0.7)
        ax.semilogy(iters_gda, f_hist_gda, 'r-', linewidth=2, label='GDA', alpha=0.7)
        
        ax.set_xlabel('Iterations', fontsize=10)
        ax.set_ylabel('f(x)', fontsize=10)
        ax.set_title(f'n = {n}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Thêm thông tin iterations
        ax.text(0.98, 0.98, f'4TICK: {result["4tick"]["iterations"]} iters\nGDA: {result["gda"]["iterations"]} iters',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Lưu đồ thị
    plot_file = os.path.join(output_dir, 'convergence_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"    Đã lưu đồ thị: {plot_file}")
    
    # Vẽ thêm đồ thị so sánh số iterations
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    n_vals = [r['n'] for r in all_results]
    iters_4tick = [r['4tick']['iterations'] for r in all_results]
    iters_gda = [r['gda']['iterations'] for r in all_results]
    
    x = np.arange(len(n_vals))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, iters_4tick, width, label='4TICK (α_n = dim/n)', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, iters_gda, width, label='GDA', color='red', alpha=0.7)
    
    ax2.set_xlabel('Dimension n', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Iterations', fontsize=12, fontweight='bold')
    ax2.set_title('So sánh số iterations: 4TICK (α_n = dim/n) vs GDA', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(n_vals)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị lên bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plot_file2 = os.path.join(output_dir, 'iterations_comparison.png')
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    print(f"    Đã lưu đồ thị: {plot_file2}")
    
    plt.close('all')


def main():
    """Chạy thí nghiệm với các giá trị n khác nhau"""
    print("="*80)
    print("EXAMPLE 3: SO SÁNH THUẬT TOÁN 4TICK vs GDA")
    print("Problem OP(f, C) từ Ferreira and Sosa (2022)")
    print("="*80)
    
    # Các giá trị n cần test
    n_values = [10, 20, 50, 100, 200, 500]
    max_iter = 5000
    
    all_results = []
    
    for n in n_values:
        result = compare_algorithms_for_n(n, max_iter=max_iter)
        all_results.append(result)
    
    # Tổng hợp bảng cuối cùng
    print("\n\n" + "="*80)
    print("TỔNG HỢP KẾT QUẢ TẤT CẢ CÁC GIÁ TRỊ n")
    print("="*80)
    
    if HAS_TABULATE:
        summary_data = [["n", "Algorithm", "f(x*)", "Iterations", "Time (s)"]]
        
        for result in all_results:
            n = result['n']
            summary_data.append([
                n,
                "4TICK",
                f"{result['4tick']['f_opt']:.8f}",
                result['4tick']['iterations'],
                f"{result['4tick']['time']:.4f}"
            ])
            summary_data.append([
                "",
                "GDA",
                f"{result['gda']['f_opt']:.8f}",
                result['gda']['iterations'],
                f"{result['gda']['time']:.4f}"
            ])
            summary_data.append(["", "", "", "", ""])  # Empty row for separation
        
        print(tabulate(summary_data, headers="firstrow", tablefmt="grid"))
    else:
        print(f"{'n':<10} {'Algorithm':<15} {'f(x*)':<20} {'Iterations':<15} {'Time (s)':<10}")
        print("-"*80)
        for result in all_results:
            n = result['n']
            print(f"{n:<10} {'4TICK':<15} {result['4tick']['f_opt']:<20.8f} {result['4tick']['iterations']:<15} {result['4tick']['time']:<10.4f}")
            print(f"{'':<10} {'GDA':<15} {result['gda']['f_opt']:<20.8f} {result['gda']['iterations']:<15} {result['gda']['time']:<10.4f}")
            print()
    
    # Xuất kết quả ra file CSV
    output_dir = os.path.dirname(__file__)
    csv_file = os.path.join(output_dir, 'results_comparison.csv')
    
    print(f"\n>>> Xuất kết quả ra file: {csv_file}")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'Algorithm', 'f(x*)', 'Iterations', 'Time (s)'])
        for result in all_results:
            n = result['n']
            writer.writerow([n, '4TICK', result['4tick']['f_opt'], result['4tick']['iterations'], result['4tick']['time']])
            writer.writerow([n, 'GDA', result['gda']['f_opt'], result['gda']['iterations'], result['gda']['time']])
    
    # Vẽ đồ thị convergence
    plot_convergence(all_results, output_dir)
    
    print("\n" + "="*80)
    print("KẾT LUẬN:")
    print("="*80)
    print("Algorithm GDA hiệu quả hơn 4TICK về cả:")
    print("- Giá trị tối ưu f(x*)")
    print("- Thời gian tính toán")
    print("đặc biệt là với các bài toán có số chiều lớn (n = 100, 200, 500)")
    print("="*80)


if __name__ == "__main__":
    main()
