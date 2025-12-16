"""
Ternary Search Algorithm for Line Search
Finds the optimal step size that minimizes a unimodal function
"""

import numpy as np
from typing import Callable, Tuple


def ternary_search_line_search(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    d: np.ndarray,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    epsilon: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, int]:
    """
    Ternary search để tìm step size tối ưu cho line search
    
    Parameters:
    -----------
    f : Callable
        Hàm mục tiêu cần tối thiểu hóa
    x : np.ndarray
        Điểm hiện tại
    d : np.ndarray
        Hướng tìm kiếm (search direction)
    alpha_min : float
        Giới hạn dưới của step size (mặc định 0.0)
    alpha_max : float
        Giới hạn trên của step size (mặc định 1.0)
    epsilon : float
        Độ chính xác (mặc định 1e-6)
    max_iter : int
        Số vòng lặp tối đa (mặc định 100)
    
    Returns:
    --------
    alpha_opt : float
        Step size tối ưu
    iterations : int
        Số vòng lặp đã thực hiện
    
    Notes:
    ------
    Ternary search hoạt động trên các hàm unimodal (hàm có duy nhất một cực tiểu
    trong khoảng tìm kiếm). Thuật toán chia khoảng tìm kiếm thành 3 phần và loại
    bỏ 1/3 khoảng ở mỗi bước lặp.
    """
    a = alpha_min
    b = alpha_max
    iterations = 0
    
    while (b - a) > epsilon and iterations < max_iter:
        # Chia khoảng [a, b] thành 3 phần bằng nhau
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        
        # Tính giá trị hàm mục tiêu tại các điểm m1 và m2
        f_m1 = f(x + m1 * d)
        f_m2 = f(x + m2 * d)
        
        # So sánh và loại bỏ 1/3 khoảng tìm kiếm
        if f_m1 > f_m2:
            # Cực tiểu nằm trong khoảng [m1, b]
            a = m1
        else:
            # Cực tiểu nằm trong khoảng [a, m2]
            b = m2
        
        iterations += 1
    
    # Step size tối ưu là điểm giữa của khoảng cuối cùng
    alpha_opt = (a + b) / 2
    
    return alpha_opt, iterations


def ternary_search_exact(
    phi: Callable[[float], float],
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    epsilon: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, int, list]:
    """
    Ternary search trực tiếp trên hàm một biến phi(alpha)
    
    Parameters:
    -----------
    phi : Callable
        Hàm mục tiêu một biến phi(alpha) = f(x + alpha * d)
    alpha_min : float
        Giới hạn dưới của step size
    alpha_max : float
        Giới hạn trên của step size
    epsilon : float
        Độ chính xác
    max_iter : int
        Số vòng lặp tối đa
    
    Returns:
    --------
    alpha_opt : float
        Step size tối ưu
    iterations : int
        Số vòng lặp đã thực hiện
    history : list
        Lịch sử các khoảng tìm kiếm [(a, b), ...]
    """
    a = alpha_min
    b = alpha_max
    iterations = 0
    history = [(a, b)]
    
    while (b - a) > epsilon and iterations < max_iter:
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        
        if phi(m1) > phi(m2):
            a = m1
        else:
            b = m2
        
        iterations += 1
        history.append((a, b))
    
    alpha_opt = (a + b) / 2
    
    return alpha_opt, iterations, history


def backtracking_with_ternary(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    d: np.ndarray,
    alpha_init: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    epsilon: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, int]:
    """
    Kết hợp backtracking line search với ternary search
    
    Parameters:
    -----------
    f : Callable
        Hàm mục tiêu
    grad_f : Callable
        Gradient của hàm mục tiêu
    x : np.ndarray
        Điểm hiện tại
    d : np.ndarray
        Hướng tìm kiếm
    alpha_init : float
        Step size khởi tạo
    rho : float
        Hệ số co (0 < rho < 1)
    c : float
        Hệ số Armijo (0 < c < 1)
    epsilon : float
        Độ chính xác cho ternary search
    max_iter : int
        Số vòng lặp tối đa cho ternary search
    
    Returns:
    --------
    alpha_opt : float
        Step size tối ưu
    iterations : int
        Số vòng lặp đã thực hiện
    """
    # Bước 1: Sử dụng backtracking để tìm khoảng chấp nhận được
    alpha = alpha_init
    f_x = f(x)
    grad_f_x = grad_f(x)
    
    # Điều kiện Armijo
    while f(x + alpha * d) > f_x + c * alpha * np.dot(grad_f_x, d):
        alpha = rho * alpha
        if alpha < 1e-10:
            break
    
    # Bước 2: Sử dụng ternary search trong khoảng [0, alpha]
    alpha_opt, iterations = ternary_search_line_search(
        f, x, d, 
        alpha_min=0.0, 
        alpha_max=alpha, 
        epsilon=epsilon, 
        max_iter=max_iter
    )
    
    return alpha_opt, iterations


# ===== Ví dụ sử dụng =====

def example_quadratic():
    """
    Ví dụ: Tìm step size tối ưu cho hàm bậc hai
    f(x) = x1^2 + x2^2
    """
    print("=" * 60)
    print("VÍ DỤ 1: HÀM BẬC HAI")
    print("=" * 60)
    
    # Định nghĩa hàm mục tiêu
    def f(x):
        return x[0]**2 + x[1]**2
    
    # Điểm xuất phát và hướng tìm kiếm
    x = np.array([2.0, 3.0])
    d = np.array([-1.0, -1.0])  # Hướng gradient descent
    
    print(f"Điểm xuất phát: x = {x}")
    print(f"Hướng tìm kiếm: d = {d}")
    print(f"Giá trị hàm tại x: f(x) = {f(x):.6f}")
    
    # Áp dụng ternary search
    alpha_opt, iterations = ternary_search_line_search(
        f, x, d, alpha_min=0.0, alpha_max=2.0, epsilon=1e-6
    )
    
    print(f"\nKết quả:")
    print(f"Step size tối ưu: α* = {alpha_opt:.6f}")
    print(f"Số vòng lặp: {iterations}")
    print(f"Điểm mới: x_new = {x + alpha_opt * d}")
    print(f"Giá trị hàm tại x_new: f(x_new) = {f(x + alpha_opt * d):.6f}")
    print()


def example_rosenbrock():
    """
    Ví dụ: Tìm step size cho hàm Rosenbrock
    f(x1, x2) = (1 - x1)^2 + 100(x2 - x1^2)^2
    """
    print("=" * 60)
    print("VÍ DỤ 2: HÀM ROSENBROCK")
    print("=" * 60)
    
    def f(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def grad_f(x):
        g1 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        g2 = 200 * (x[1] - x[0]**2)
        return np.array([g1, g2])
    
    x = np.array([0.0, 0.0])
    g = grad_f(x)
    d = -g / np.linalg.norm(g)  # Hướng gradient descent chuẩn hóa
    
    print(f"Điểm xuất phát: x = {x}")
    print(f"Gradient: ∇f(x) = {g}")
    print(f"Hướng tìm kiếm: d = {d}")
    print(f"Giá trị hàm tại x: f(x) = {f(x):.6f}")
    
    # Phương pháp 1: Ternary search thuần túy
    alpha_opt1, iterations1 = ternary_search_line_search(
        f, x, d, alpha_min=0.0, alpha_max=1.0, epsilon=1e-6
    )
    
    print(f"\nPhương pháp 1: Ternary Search")
    print(f"Step size tối ưu: α* = {alpha_opt1:.6f}")
    print(f"Số vòng lặp: {iterations1}")
    print(f"f(x + α*d) = {f(x + alpha_opt1 * d):.6f}")
    
    # Phương pháp 2: Kết hợp backtracking và ternary search
    alpha_opt2, iterations2 = backtracking_with_ternary(
        f, grad_f, x, d, alpha_init=1.0, epsilon=1e-6
    )
    
    print(f"\nPhương pháp 2: Backtracking + Ternary Search")
    print(f"Step size tối ưu: α* = {alpha_opt2:.6f}")
    print(f"Số vòng lặp: {iterations2}")
    print(f"f(x + α*d) = {f(x + alpha_opt2 * d):.6f}")
    print()


def example_comparison():
    """
    So sánh tốc độ hội tụ của ternary search
    """
    print("=" * 60)
    print("VÍ DỤ 3: SO SÁNH TỐC ĐỘ HỘI TỤ")
    print("=" * 60)
    
    def phi(alpha):
        return alpha**2 - 4*alpha + 5
    
    print("Hàm mục tiêu: φ(α) = α² - 4α + 5")
    print("Cực tiểu lý thuyết tại α* = 2")
    print()
    
    # Chạy với các epsilon khác nhau
    epsilons = [1e-3, 1e-6, 1e-9]
    
    for eps in epsilons:
        alpha_opt, iterations, history = ternary_search_exact(
            phi, alpha_min=0.0, alpha_max=4.0, epsilon=eps
        )
        print(f"Epsilon = {eps:.0e}:")
        print(f"  α* = {alpha_opt:.10f}")
        print(f"  φ(α*) = {phi(alpha_opt):.10f}")
        print(f"  Số vòng lặp: {iterations}")
        print(f"  Sai số: |α* - 2| = {abs(alpha_opt - 2):.2e}")
        print()


if __name__ == "__main__":
    # Chạy các ví dụ
    example_quadratic()
    example_rosenbrock()
    example_comparison()
