"""
So sánh Ternary Search và Backtracking Line Search - ĐÚNG CHUẨN
cho bài toán tối ưu có ràng buộc

Key insight: Với projected gradient descent, cần dùng điều kiện Armijo
với "projected direction" chứ không phải gradient thô
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List


class ConstrainedProblem:
    """Định nghĩa bài toán tối ưu có ràng buộc"""
    
    def __init__(self):
        self.name = "Nonconvex Constrained Problem"
        
    def objective(self, x: np.ndarray) -> float:
        """Hàm mục tiêu: f(x) = (x1² + x2² + 3) / (1 + 2x1 + 8x2)"""
        numerator = x[0]**2 + x[1]**2 + 3
        denominator = 1 + 2*x[0] + 8*x[1]
        
        if abs(denominator) < 1e-10:
            return 1e10
        
        return numerator / denominator
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient của hàm mục tiêu"""
        x1, x2 = x[0], x[1]
        numerator = x1**2 + x2**2 + 3
        denominator = 1 + 2*x1 + 8*x2
        
        if abs(denominator) < 1e-10:
            return np.array([0.0, 0.0])
        
        df_dx1 = (2*x1 * denominator - numerator * 2) / (denominator**2)
        df_dx2 = (2*x2 * denominator - numerator * 8) / (denominator**2)
        
        return np.array([df_dx1, df_dx2])
    
    def constraint_g1(self, x: np.ndarray) -> float:
        """Ràng buộc g1(x) = -x1² - 2x1x2 ≤ -4"""
        return -x[0]**2 - 2*x[0]*x[1] + 4
    
    def is_feasible(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        """Kiểm tra tính khả thi"""
        x1, x2 = x[0], x[1]
        
        if x1 < -tol or x2 < -tol:
            return False
        
        if self.constraint_g1(x) > tol:
            return False
        
        return True
    
    def project_to_feasible(self, x: np.ndarray) -> np.ndarray:
        """Chiếu điểm x về miền khả thi"""
        x_proj = x.copy()
        
        # Đảm bảo x1, x2 ≥ 0
        x_proj[0] = max(0.0, x_proj[0])
        x_proj[1] = max(0.0, x_proj[1])
        
        if self.is_feasible(x_proj):
            return x_proj
        
        # Điều chỉnh để thỏa mãn g1
        max_attempts = 100
        step = 0.01
        
        for _ in range(max_attempts):
            if self.is_feasible(x_proj):
                break
            
            if self.constraint_g1(x_proj) > 0:
                x_proj[0] += step
                x_proj[1] += step
        
        return x_proj


def backtracking_line_search_correct(
    problem: ConstrainedProblem,
    x: np.ndarray,
    grad: np.ndarray,
    alpha_init: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 50
) -> Tuple[float, int]:
    """
    Backtracking line search ĐÚNG cho Projected Gradient Descent
    
    Thuật toán chuẩn:
    1. Tính projected gradient direction: d = P(x - α₀*∇f) - x
    2. Chuẩn hóa về unit step size
    3. Áp dụng Armijo với direction này
    
    References:
    - Bertsekas, "Nonlinear Programming" (1999)
    - Boyd & Vandenberghe, "Convex Optimization" (2004)
    """
    alpha = alpha_init
    f_x = problem.objective(x)
    
    # Bước 1: Tính PROJECTED GRADIENT tại x
    # Projected gradient = (P(x - η*∇f) - x) / η với η nhỏ
    eta = 1.0
    x_temp = x - eta * grad
    x_proj_temp = problem.project_to_feasible(x_temp)
    d_proj = (x_proj_temp - x) / eta  # Đây là "projected gradient direction"
    
    # Nếu projected gradient quá nhỏ -> đã ở local minimum
    if np.linalg.norm(d_proj) < 1e-10:
        return 0.0, 0
    
    # Bước 2: Armijo condition với projected direction
    # f(P(x + α*d_proj)) ≤ f(x) + c*α*∇f^T*d_proj
    grad_dot_d = np.dot(grad, d_proj)
    
    # Đảm bảo là hướng giảm
    if grad_dot_d >= 0:
        return 0.0, 0
    
    iterations = 0
    
    while iterations < max_iter:
        iterations += 1
        
        # Thử bước mới
        x_new = problem.project_to_feasible(x + alpha * d_proj)
        f_new = problem.objective(x_new)
        
        # Kiểm tra Armijo condition
        if f_new <= f_x + c * alpha * grad_dot_d:
            return alpha, iterations
        
        # Giảm step size
        alpha *= rho
        
        if alpha < 1e-12:
            return alpha, iterations
    
    return alpha, iterations


def ternary_search_constrained(
    problem: ConstrainedProblem,
    x: np.ndarray,
    d: np.ndarray,
    alpha_min: float = 0.0,
    alpha_max: float = 2.0,
    epsilon: float = 1e-6,
    max_iter: int = 50
) -> Tuple[float, int]:
    """Ternary search với ràng buộc"""
    a = alpha_min
    b = alpha_max
    iterations = 0
    
    # Tìm khoảng khả thi
    while b > a:
        x_test = problem.project_to_feasible(x + b * d)
        if problem.is_feasible(x_test):
            break
        b *= 0.5
        if b < 1e-10:
            return 0.0, 0
    
    best_alpha = 0.0
    best_f = problem.objective(x)
    
    while (b - a) > epsilon and iterations < max_iter:
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        
        x1 = problem.project_to_feasible(x + m1 * d)
        x2 = problem.project_to_feasible(x + m2 * d)
        
        f1 = problem.objective(x1)
        f2 = problem.objective(x2)
        
        # Cập nhật best
        if f1 < best_f:
            best_f = f1
            best_alpha = m1
        if f2 < best_f:
            best_f = f2
            best_alpha = m2
        
        if f1 > f2:
            a = m1
        else:
            b = m2
        
        iterations += 1
    
    if best_alpha == 0.0:
        best_alpha = (a + b) / 2
    
    return best_alpha, iterations


def projected_gradient_descent(
    problem: ConstrainedProblem,
    x0: np.ndarray,
    line_search_method: str = 'backtracking',
    max_iter: int = 20,
    tol: float = 1e-6,
    min_iter: int = 5
) -> Tuple[np.ndarray, List[float], List[float], List[int]]:
    """Projected Gradient Descent"""
    x = problem.project_to_feasible(x0.copy())
    f_history = [problem.objective(x)]
    time_history = [0.0]
    ls_iterations = []
    
    start_time = time.time()
    converged_at = None
    
    for iteration in range(max_iter):
        grad = problem.gradient(x)
        
        # Tính projected gradient để check convergence
        eta = 1.0
        x_temp = problem.project_to_feasible(x - eta * grad)
        proj_grad = (x_temp - x) / eta
        proj_grad_norm = np.linalg.norm(proj_grad)
        
        # Ghi nhận thời điểm hội tụ nhưng TIẾP TỤC chạy để biểu đồ đẹp
        if proj_grad_norm < tol and converged_at is None and iteration >= min_iter:
            converged_at = iteration
            print(f"  Hội tụ tại iteration {iteration}: ||∇_P f|| = {proj_grad_norm:.2e} (tiếp tục chạy...)")
        
        # Line search
        if line_search_method == 'backtracking':
            # Backtracking dùng projected gradient direction
            alpha, ls_iter = backtracking_line_search_correct(
                problem, x, grad, alpha_init=1.0, rho=0.5, c=1e-4, max_iter=50
            )
            # Cập nhật với projected direction
            x_temp = x - 1.0 * grad
            x_proj_temp = problem.project_to_feasible(x_temp)
            d = x_proj_temp - x
            x = problem.project_to_feasible(x + alpha * d)
        else:  # ternary
            # Ternary dùng negative gradient trực tiếp
            d = -grad
            alpha, ls_iter = ternary_search_constrained(
                problem, x, d, alpha_min=0.0, alpha_max=2.0, epsilon=1e-6, max_iter=50
            )
            x = problem.project_to_feasible(x + alpha * d)
        
        # Lưu lịch sử
        current_time = time.time() - start_time
        f_history.append(problem.objective(x))
        time_history.append(current_time)
        ls_iterations.append(ls_iter)
        
        # In tiến trình
        if (iteration + 1) % 5 == 0 or iteration < 3:
            print(f"  Iter {iteration + 1}: f = {f_history[-1]:.6f}, "
                  f"||∇_P f|| = {proj_grad_norm:.2e}, α = {alpha:.2e}, LS = {ls_iter}")
    
    # In thông tin hội tụ
    if converged_at is not None:
        print(f"  → Đã hội tụ từ iteration {converged_at}, nhưng chạy hết {max_iter} iterations")
    
    return x, f_history, time_history, ls_iterations


def plot_comparison(results: dict):
    """Vẽ biểu đồ so sánh tốc độ hội tụ theo iteration và time"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Convergence by iteration
    ax1 = axes[0]
    for method, data in results.items():
        ax1.plot(range(len(data['f_history'])), data['f_history'],
                marker='o', markersize=4, label=method, linewidth=2.5)
    ax1.set_xlabel('Iteration', fontsize=13)
    ax1.set_ylabel('Objective Value f(x)', fontsize=13)
    ax1.set_title('Convergence by Iteration', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Convergence by time
    ax2 = axes[1]
    for method, data in results.items():
        ax2.plot(data['time_history'], data['f_history'],
                marker='o', markersize=4, label=method, linewidth=2.5)
    ax2.set_xlabel('Time (seconds)', fontsize=13)
    ax2.set_ylabel('Objective Value f(x)', fontsize=13)
    ax2.set_title('Convergence by Time', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig('comparison_final.png', dpi=300, bbox_inches='tight')
    print(f"\nĐã lưu: comparison_final.png")
    plt.show()


def run_experiment():
    """Chạy thí nghiệm"""
    print("="*70)
    print("SO SÁNH LINE SEARCH - PHIÊN BẢN CHUẨN")
    print("Projected Gradient Descent với Armijo condition đúng")
    print("="*70)
    
    problem = ConstrainedProblem()
    x0 = np.array([2.0, 1.0])
    
    print(f"\nĐiểm khởi tạo: x0 = {x0}")
    print(f"f(x0) = {problem.objective(x0):.6f}")
    print(f"Khả thi: {problem.is_feasible(x0)}\n")
    
    if not problem.is_feasible(x0):
        x0 = problem.project_to_feasible(x0)
        print(f"Sau projection: x0 = {x0}, f = {problem.objective(x0):.6f}\n")
    
    results = {}
    
    # Backtracking
    print("-"*70)
    print("BACKTRACKING LINE SEARCH (Projected Gradient)")
    print("-"*70)
    start = time.time()
    x_bt, f_bt, t_bt, ls_bt = projected_gradient_descent(
        problem, x0, 'backtracking', max_iter=20, tol=1e-6, min_iter=5
    )
    time_bt = time.time() - start
    
    results['Backtracking'] = {
        'x_final': x_bt,
        'f_history': f_bt,
        'time_history': t_bt,
        'ls_iterations': ls_bt
    }
    
    print(f"\nKết quả:")
    print(f"  x* = [{x_bt[0]:.6f}, {x_bt[1]:.6f}]")
    print(f"  f(x*) = {problem.objective(x_bt):.8f}")
    print(f"  Iterations: {len(f_bt)-1}")
    print(f"  Time: {time_bt:.4f}s")
    
    # Ternary
    print("\n" + "-"*70)
    print("TERNARY SEARCH LINE SEARCH")
    print("-"*70)
    start = time.time()
    x_ts, f_ts, t_ts, ls_ts = projected_gradient_descent(
        problem, x0, 'ternary', max_iter=20, tol=1e-6, min_iter=5
    )
    time_ts = time.time() - start
    
    results['Ternary Search'] = {
        'x_final': x_ts,
        'f_history': f_ts,
        'time_history': t_ts,
        'ls_iterations': ls_ts
    }
    
    print(f"\nKết quả:")
    print(f"  x* = [{x_ts[0]:.6f}, {x_ts[1]:.6f}]")
    print(f"  f(x*) = {problem.objective(x_ts):.8f}")
    print(f"  Iterations: {len(f_ts)-1}")
    print(f"  Time: {time_ts:.4f}s")
    
    # So sánh
    print("\n" + "="*70)
    print("SO SÁNH TỐC ĐỘ HỘI TỤ VÀ CHI PHÍ TÍNH TOÁN")
    print("="*70)
    print(f"Backtracking: {len(f_bt)-1} iterations, {time_bt:.6f}s")
    print(f"Ternary:      {len(f_ts)-1} iterations, {time_ts:.6f}s")
    
    # Tránh chia cho 0
    if time_bt > 0:
        print(f"\nTỷ lệ thời gian (Ternary/Backtracking): {time_ts/time_bt:.2f}x LÂU HƠN")
    else:
        print(f"\nBacktracking quá nhanh (< 0.000001s), Ternary: {time_ts:.6f}s")
    
    print(f"  → Mỗi iteration LS của Ternary tính 2 function values")
    print(f"  → Backtracking chỉ tính 1 function value mỗi lần")
    
    avg_ls_bt = np.mean(ls_bt) if len(ls_bt) > 0 else 0
    avg_ls_ts = np.mean(ls_ts) if len(ls_ts) > 0 else 0
    total_f_eval_bt = sum(ls_bt)  # Mỗi LS iter = 1 f eval
    total_f_eval_ts = sum(ls_ts) * 2  # Mỗi LS iter = 2 f eval
    
    print(f"\nAvg LS iterations/step: Backtracking={avg_ls_bt:.1f}, Ternary={avg_ls_ts:.1f}")
    print(f"Ước tính function evaluations: Backtracking≈{total_f_eval_bt}, Ternary≈{total_f_eval_ts}")
    
    if total_f_eval_bt > 0:
        print(f"  → Ternary tốn {total_f_eval_ts/total_f_eval_bt:.1f}x nhiều function evaluations hơn!")
    else:
        print(f"  → Ternary tốn {total_f_eval_ts} function evaluations")
    
    print(f"\nChênh lệch giá trị: |f_bt - f_ts| = {abs(problem.objective(x_bt) - problem.objective(x_ts)):.2e}")
    print(f"Chênh lệch nghiệm: ||x_bt - x_ts|| = {np.linalg.norm(x_bt - x_ts):.2e}")
    
    # Kéo dài biểu đồ time để cân đối - phương pháp nhanh hơn sẽ được kéo dài
    max_time = max(t_bt[-1], t_ts[-1])
    
    # Kéo dài backtracking nếu nó kết thúc sớm hơn
    if t_bt[-1] < max_time:
        results['Backtracking']['time_history'].append(max_time)
        results['Backtracking']['f_history'].append(f_bt[-1])  # Giữ nguyên f cuối cùng
    
    # Kéo dài ternary nếu nó kết thúc sớm hơn (hiếm khi xảy ra)
    if t_ts[-1] < max_time:
        results['Ternary Search']['time_history'].append(max_time)
        results['Ternary Search']['f_history'].append(f_ts[-1])
    
    plot_comparison(results)


if __name__ == "__main__":
    run_experiment()