"""
So sánh Projected Gradient (α_n = 1/n) vs GDA
Bài toán gốc với tập ràng buộc C (không có D)

Bài toán:
minimize f(x) = (x1² + x2² + 3) / (1 + 2x1 + 8x2)
subject to x ∈ C

Với C = {x = (x1, x2)^T ∈ R² | x1² + 2x1x2 ≥ 4; x1, x2 ≥ 0}
"""

from random import random
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

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
sys.modules['tick4'] = tick4_module  # Force reload
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


class Problem:
    """Định nghĩa bài toán"""
    
    def __init__(self):
        self.name = "Constrained Optimization on C"
    
    # Numpy version (cho 4TICK)
    def objective_np(self, x: np.ndarray) -> float:
        """f(x) = (x1² + x2² + 3) / (1 + 2x1 + 8x2)"""
        numerator = x[0]**2 + x[1]**2 + 3
        denominator = 1 + 2*x[0] + 8*x[1]
        if abs(denominator) < 1e-10:
            return 1e10
        return numerator / denominator
    
    def gradient_np(self, x: np.ndarray) -> np.ndarray:
        """Gradient của f"""
        x1, x2 = x[0], x[1]
        numerator = x1**2 + x2**2 + 3
        denominator = 1 + 2*x1 + 8*x2
        
        if abs(denominator) < 1e-10:
            return np.array([0.0, 0.0])
        
        df_dx1 = (2*x1 * denominator - numerator * 2) / (denominator**2)
        df_dx2 = (2*x2 * denominator - numerator * 8) / (denominator**2)
        
        return np.array([df_dx1, df_dx2])
    
    def projection_np(self, x: np.ndarray) -> np.ndarray:
        """Chiếu lên C: {x ≥ 0, x1² + 2x1x2 ≥ 4}"""
        x_proj = x.copy()
        
        # Bước 1: x ≥ 0
        x_proj[0] = max(0.0, x_proj[0])
        x_proj[1] = max(0.0, x_proj[1])
        
        # Bước 2: Ràng buộc phi tuyến x1² + 2x1x2 ≥ 4
        max_iter = 20
        for _ in range(max_iter):
            x1, x2 = x_proj[0], x_proj[1]
            constraint_val = x1**2 + 2*x1*x2 - 4
            
            if constraint_val >= -1e-6:
                break
            
            # Gradient của ràng buộc
            grad_c = np.array([2*x1 + 2*x2, 2*x1])
            norm_grad_sq = np.sum(grad_c**2)
            
            if norm_grad_sq < 1e-6:
                break
            
            # Newton-like update
            step = -(constraint_val / norm_grad_sq) * grad_c
            x_proj = x_proj + step
            
            # Đảm bảo x ≥ 0
            x_proj[0] = max(0.0, x_proj[0])
            x_proj[1] = max(0.0, x_proj[1])
        
        return x_proj
    
    # PyTorch version (cho GDA)
    def objective_torch(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float64)
        x1, x2 = x[0], x[1]
        return (x1**2 + x2**2 + 3) / (1 + 2*x1 + 8*x2)
    
    def gradient_torch(self, x: torch.Tensor) -> torch.Tensor:
        xv = x.clone().detach().requires_grad_(True)
        f = self.objective_torch(xv)
        f.backward()
        return xv.grad.detach()
    
    def projection_torch(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(torch.float64)
            
            # x ≥ 0
            x = torch.maximum(x, torch.tensor(0.0, dtype=torch.float64))
            
            # x1² + 2x1x2 ≥ 4
            for _ in range(20):
                x1, x2 = x[0], x[1]
                constraint_val = x1**2 + 2*x1*x2 - 4
                
                if constraint_val >= -1e-6:
                    break
                
                grad_c = torch.tensor([2*x1 + 2*x2, 2*x1], dtype=torch.float64)
                norm_grad_sq = torch.sum(grad_c**2)
                
                if norm_grad_sq < 1e-6:
                    break
                
                step = -(constraint_val / norm_grad_sq) * grad_c
                x = x + step
                x = torch.maximum(x, torch.tensor(0.0, dtype=torch.float64))
        
        return x


def run_4tick_method(problem: Problem, x0: np.ndarray, max_iter: int = 1000, 
                     alpha_func=None, method_name="4TICK"):
    """Chạy phương pháp Projected Gradient với dãy α_n tùy chỉnh"""
    if alpha_func is None:
        alpha_func = lambda n: 1.0 / n
        alpha_desc = "α_n = 1/n"
    else:
        alpha_desc = method_name
    
    print("="*70)
    print(f"PHƯƠNG PHÁP: PROJECTED GRADIENT - {alpha_desc}")
    print("="*70)
    
    start_time = time.time()
    
    x_opt, f_history, step_history = projected_gradient_decreasing_step(
        objective=problem.objective_np,
        gradient=problem.gradient_np,
        projection=problem.projection_np,
        x0=x0,
        max_iter=1000,
        tol=1e-8,  # Tol phù hợp để dừng sớm khi hội tụ
        alpha_func=alpha_func
    )
    
    elapsed_time = time.time() - start_time
    
    actual_iters = len(f_history) - 1
    converged = actual_iters < max_iter
    
    print(f"\nKết quả:")
    print(f"  x* = [{x_opt[0]:.6f}, {x_opt[1]:.6f}]")
    print(f"  f(x*) = {problem.objective_np(x_opt):.8f}")
    print(f"  Iterations: {actual_iters}/{max_iter} {'(HỘI TỤ SỚM!)' if converged else ''}")
    print(f"  Time: {elapsed_time:.4f}s")
    
    return {
        'x_final': x_opt,
        'f_history': f_history,
        'iterations': len(f_history) - 1,
        'time': elapsed_time,
        'method': method_name
    }


def run_gda_method(problem: Problem, x0: np.ndarray, max_iter: int = 1000):
    """Chạy phương pháp GDA"""
    print("\n" + "="*70)
    print("PHƯƠNG PHÁP: GDA")
    print("="*70)
    
    # Chuyển sang PyTorch
    x0_torch = torch.tensor(x0, dtype=torch.float64)
    
    # Lưu lịch sử
    f_history = []
    
    # Wrapper để lưu lịch sử
    original_obj = problem.objective_torch
    def obj_with_history(x):
        f = original_obj(x)
        f_history.append(float(f))
        return f
    
    start_time = time.time()
    
    x_opt = run_gda_solve(
        grad_func=problem.gradient_torch,
        proj_func=problem.projection_torch,
        obj_func=obj_with_history,
        x0=x0_torch,
        step_size=1.0,
        sigma=0.1,
        kappa=0.5,
        max_iter=1000,
        tol=1e-5  # Tol phù hợp để dừng sớm khi hội tụ
    )
    
    elapsed_time = time.time() - start_time
    
    x_opt_np = x_opt.cpu().numpy()
    
    actual_iters = len(f_history)
    converged = actual_iters < max_iter
    
    print(f"\nKết quả:")
    print(f"  x* = [{x_opt_np[0]:.6f}, {x_opt_np[1]:.6f}]")
    print(f"  f(x*) = {float(original_obj(x_opt)):.8f}")
    print(f"  Iterations: {actual_iters}/{max_iter} {'(HỘI TỤ SỚM!)' if converged else ''}")
    print(f"  Time: {elapsed_time:.4f}s")
    
    return {
        'x_final': x_opt_np,
        'f_history': f_history,
        'iterations': len(f_history),
        'time': elapsed_time,
        'method': 'GDA'
    }


def plot_comparison(results: dict):
    """Vẽ biểu đồ so sánh"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Convergence by iteration
    ax1 = axes[0]
    colors = {'GDA': 'blue', '4TICK: 1/n': 'orange', '4TICK: 5/n': 'green', '4TICK: 1/(n+10)': 'red'}
    for method, data in results.items():
        iterations = range(len(data['f_history']))
        color = colors.get(method, None)
        ax1.plot(iterations, data['f_history'],
                marker='o' if len(data['f_history']) < 100 else None, 
                markersize=2, label=data['method'], linewidth=2.5, color=color)
    
    ax1.set_xlabel('Iteration', fontsize=13)
    ax1.set_ylabel('Objective Value f(x)', fontsize=13)
    ax1.set_title('Convergence Comparison', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Summary table
    ax2 = axes[1]
    ax2.axis('off')
    
    summary = "SO SÁNH KET QUA\n" + "="*60 + "\n\n"
    
    for method, data in results.items():
        summary += f"{data['method']}:\n"
        summary += f"  x* = [{data['x_final'][0]:.6f}, {data['x_final'][1]:.6f}]\n"
        summary += f"  f(x*) = {data['f_history'][-1]:.8f}\n"
        summary += f"  Iterations: {data['iterations']}\n"
        summary += f"  Time: {data['time']:.4f}s\n\n"
    
    # Tìm phương pháp tốt nhất
    best_method = min(results.keys(), key=lambda k: results[k]['f_history'][-1])
    fastest_method = min(results.keys(), key=lambda k: results[k]['time'])
    
    summary += "="*60 + "\n"
    summary += f"Phương pháp TOT NHAT (f min): {best_method}\n"
    summary += f"  f* = {results[best_method]['f_history'][-1]:.8f}\n"
    
    ax2.text(0.1, 0.9, summary, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('comparison_4tick_gda.png', dpi=300, bbox_inches='tight')
    print(f"\nĐã lưu biểu đồ: comparison_4tick_gda.png")
    plt.show()


def main():
    print("="*70)
    print("SO SÁNH: PROJECTED GRADIENT (α_n=1/n) vs GDA")
    print("Bài toán tối ưu trên tập C")
    print("="*70)
    
    problem = Problem()
    np.random.seed(36)  # Seed cố định để tái hiện kết quả
    x0 = np.random.uniform(0, 3, size=2)
    
    print(f"\nBài toán:")
    print(f"  minimize f(x) = (x₁² + x₂² + 3) / (1 + 2x₁ + 8x₂)")
    print(f"  subject to x ∈ C = {{x₁² + 2x₁x₂ ≥ 4, x₁ ≥ 0, x₂ ≥ 0}}")
    print(f"\nĐiểm khởi tạo: x0 = {x0}")
    print(f"f(x0) = {problem.objective_np(x0):.6f}\n")
    
    results = {}
    
    # Chạy GDA
    results['GDA'] = run_gda_method(problem, x0, max_iter=1000)
    
    # Chạy 4TICK với các dãy α_n khác nhau
    print("\n" + "="*70)
    print("THỬ NGHIỆM CÁC DÃY α_n KHÁC NHAU")
    print("="*70)
    
    # 1. α_n = 1/n (chuẩn)
    results['4TICK: 1/n'] = run_4tick_method(
        problem, x0, max_iter=1000,
        alpha_func=lambda n: 1.0 / n,
        method_name="4TICK: α_n=1/n"
    )
    
    # 2. α_n = 5/n (bước lớn hơn)
    results['4TICK: 5/n'] = run_4tick_method(
        problem, x0, max_iter=1000,
        alpha_func=lambda n: 5.0 / n,
        method_name="4TICK: α_n=5/n"
    )
    
    # 3. α_n = 1/(n+10) (giảm chậm hơn)
    results['4TICK: 1/(n+10)'] = run_4tick_method(
        problem, x0, max_iter=1000,
        alpha_func=lambda n: 1.0 / (n + 10),
        method_name="4TICK: α_n=1/(n+10)"
    )
    
    # Vẽ biểu đồ
    print("\n" + "="*70)
    print("Đang vẽ biểu đồ so sánh...")
    plot_comparison(results)


if __name__ == "__main__":
    main()
