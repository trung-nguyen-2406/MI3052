"""
Thuật toán Projected Gradient với step size α_n = 1/n

Cho dãy bước (α_n)_n≥1 thỏa:
    Σ α_n = ∞,  Σ α_n² < ∞

Khởi tạo x¹ ∈ C. Với mỗi n ≥ 1, đặt:
    λ_n := α_n / max{1, ||∇f(x^n)||}

và:
    x^(n+1) = P_C(x^n - λ_n ∇f(x^n))

Với α_n = 1/n, điều kiện được thỏa mãn.
"""

import numpy as np
from typing import Callable, Tuple, List


def projected_gradient_decreasing_step(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    projection: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    alpha_func: Callable[[int], float] = None
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Projected Gradient Descent với step size giảm dần
    
    Parameters:
    -----------
    objective : Callable
        Hàm mục tiêu f(x)
    gradient : Callable
        Gradient ∇f(x)
    projection : Callable
        Phép chiếu P_C(x)
    x0 : np.ndarray
        Điểm khởi tạo
    max_iter : int
        Số vòng lặp tối đa
    tol : float
        Ngưỡng dừng
    alpha_func : Callable[[int], float]
        Hàm tính α_n, mặc định α_n = 1/n
    
    Returns:
    --------
    x : np.ndarray
        Điểm tối ưu
    f_history : List[float]
        Lịch sử giá trị hàm
    step_history : List[float]
        Lịch sử step size thực tế λ_n
    """
    if alpha_func is None:
        alpha_func = lambda n: 1.0 / n  # α_n = 1/n
    
    x = projection(x0.copy())
    f_history = [objective(x)]
    step_history = []
    
    for n in range(1, max_iter + 1):
        # Tính gradient
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        # Tính step size theo công thức
        alpha_n = alpha_func(n)
        lambda_n = alpha_n / max(1.0, grad_norm)
        
        # Cập nhật với projection
        x_new = x - lambda_n * grad
        x_next = projection(x_new)
        
        # Kiểm tra điều kiện dừng: ||x_next - x|| (projected gradient criterion)
        # GIỐNG ĐIỀU KIỆN CỦA GDA: torch.norm(x_new - x) < tol
        step_norm = np.linalg.norm(x_next - x)
        if step_norm < tol:
            print(f"  >> Hội tụ tại iteration {n}: ||x_next - x|| = {step_norm:.2e} < tol={tol:.2e}")
            x = x_next
            f_history.append(objective(x))
            step_history.append(lambda_n)
            break
        
        x = x_next
        
        # Lưu lịch sử
        f_val = objective(x)
        f_history.append(f_val)
        step_history.append(lambda_n)
        
        # In tiến trình
        if n % 100 == 0 or n <= 10:
            print(f"  Iter {n}: f = {f_val:.6f}, ||∇f|| = {grad_norm:.2e}, "
                  f"α_n = {alpha_n:.4f}, λ_n = {lambda_n:.4f}")
    
    return x, f_history, step_history


def projected_gradient_fixed_step(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    projection: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    step_size: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Projected Gradient Descent với step size cố định (để so sánh)
    """
    x = projection(x0.copy())
    f_history = [objective(x)]
    
    for n in range(1, max_iter + 1):
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            print(f"Hội tụ tại iteration {n}: ||∇f|| = {grad_norm:.2e}")
            break
        
        x_new = x - step_size * grad
        x = projection(x_new)
        
        f_val = objective(x)
        f_history.append(f_val)
        
        if n % 100 == 0 or n <= 10:
            print(f"  Iter {n}: f = {f_val:.6f}, ||∇f|| = {grad_norm:.2e}")
    
    return x, f_history


# Ví dụ sử dụng
if __name__ == "__main__":
    print("="*70)
    print("THUẬT TOÁN PROJECTED GRADIENT VỚI α_n = 1/n")
    print("="*70)
    
    # Ví dụ đơn giản: minimize x² + y² subject to x, y ≥ 1
    def objective(x):
        return x[0]**2 + x[1]**2
    
    def gradient(x):
        return 2 * x
    
    def projection(x):
        return np.maximum(x, 1.0)
    
    x0 = np.array([5.0, 5.0])
    
    print("\nBài toán: min x₁² + x₂² subject to x₁, x₂ ≥ 1")
    print(f"Điểm khởi tạo: x0 = {x0}")
    print(f"Nghiệm lý thuyết: x* = [1, 1], f(x*) = 2\n")
    
    print("Chạy thuật toán với α_n = 1/n:")
    x_opt, f_hist, step_hist = projected_gradient_decreasing_step(
        objective, gradient, projection, x0, max_iter=100, tol=1e-6
    )
    
    print(f"\nKết quả:")
    print(f"  x* = {x_opt}")
    print(f"  f(x*) = {objective(x_opt):.8f}")
    print(f"  Số iterations: {len(f_hist) - 1}")
    print(f"  Step size cuối cùng: {step_hist[-1]:.6f}")
