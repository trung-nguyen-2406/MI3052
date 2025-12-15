import sys, pathlib
import numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # ensure src on path
import torch
from gda import run_gda_solve

def theta_smooth(x, mu=0.1):
    """Smooth approximation of |x|"""
    if torch.abs(x) >= (mu / 2):
        return torch.abs(x)
    else:
        return x**2 / mu + mu / 4

def obj_func(x, mu=0.01):
    """Smoothed objective function"""
    x = x.to(torch.float64)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return (torch.exp(theta_smooth(x2 - 3, mu)) - 30) / (x1 ** 2 + x3 ** 2 + 2 * x4 ** 2 + 4) 

def grad_func(x):
    # compute gradient via autograd
    xv = x.clone().detach().requires_grad_(True)
    f = obj_func(xv, mu=0.01)
    f.backward()
    return xv.grad.detach()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------
# 1. Định nghĩa các Ràng buộc (Constraints)
#    Lưu ý: Chúng ta cần một cách để xử lý đẳng thức và bất đẳng thức
# -----------------------------------------------------------------

# Ràng buộc Bất đẳng thức g1(x) <= 10
# Tương đương: g1(x) - 10 <= 0
def constraint_g1(x):
    """g1(x) = (x1 + x3)^3 + 2*x2^2"""
    # x là tensor 4 chiều: [x1, x2, x3, x4]
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return (x1 + x3)**3 + 2 * x2**2 - 10.0

# Ràng buộc Bất đẳng thức g2(x) <= 1
# Tương đương: g2(x) - 1 <= 0
def constraint_g2(x):
    """g2(x) = (x2 - 1)^2"""
    x2 = x[1]
    return (x2 - 1)**2 - 1.0

# Ràng buộc Đẳng thức h(x) = 0
# h(x) = 2x1 + 4x2 + x3 + 1 = 0
def constraint_h(x):
    """h(x) = 2x1 + 4x2 + x3 + 1"""
    x1, x2, x3 = x[0], x[1], x[2]
    return 2 * x1 + 4 * x2 + x3 + 1.0

# -----------------------------------------------------------------
# 2. Hàm Chiếu (Projection Function) P_C(y)
#    Sử dụng Tối ưu hóa lồng (Nested Optimization)
# -----------------------------------------------------------------

def proj_func(y_point, max_iters=100, learning_rate=0.01):
    """
    Tính P_C(y) bằng cách giải bài toán tối ưu hóa phụ với LBFGS.
    
    Tham số:
    - y_point (torch.Tensor): Điểm y cần chiếu.
    - max_iters (int): Số lần lặp tối đa cho bài toán chiếu.
    - learning_rate (float): Tốc độ học cho bộ giải bên trong.
    
    Trả về:
    - x_star (torch.Tensor): P_C(y), điểm được chiếu.
    """
    
    # print(f"\n--- Tính P_C(y) cho y = {y_point.cpu().numpy()} ---")

    # Khởi tạo điểm x (cần tính gradient)
    x = y_point.clone().detach().to(device).requires_grad_(True)
    
    # Bộ giải: LBFGS optimizer with strong_wolfe line search
    optimizer = torch.optim.LBFGS([x], lr=learning_rate, max_iter=20, line_search_fn='strong_wolfe')
    
    # y_point không cần tính gradient, dùng .detach()
    y_target = y_point.detach()

    def closure():
        optimizer.zero_grad()
        
        # 1. Hàm mục tiêu (Objective Function): min 1/2 * ||x - y||^2
        objective_loss = 0.5 * torch.sum((x - y_target)**2)
        
        # 2. Xử lý Ràng buộc (Sử dụng Penalty Method)
        
        # Ràng buộc Đẳng thức: |h(x)|^2
        h_x = constraint_h(x)
        equality_penalty = 10000 * (h_x**2) 
        
        # Ràng buộc Bất đẳng thức: max(0, g_i(x))^2
        g1_x = constraint_g1(x)
        g2_x = constraint_g2(x)
        inequality_penalty = 10000 * (torch.max(torch.tensor(0.0).to(device), g1_x)**2 + 
                                      torch.max(torch.tensor(0.0).to(device), g2_x)**2)
        
        # Tổng tổn thất (Loss)
        total_loss = objective_loss + equality_penalty + inequality_penalty
        
        # Tính toán gradient và cập nhật
        total_loss.backward()
        return total_loss
    
    optimizer.step(closure)
    return x.detach().cpu()

def run_example():
    # Feasible starting point, but far from optimum
    x0 = torch.tensor([1.0, 1.5, -9.0, 0.5], dtype=torch.float64)
    sigma = 0.1        
    kappa = 0.5         

    x_opt = run_gda_solve(grad_func, proj_func, lambda x: obj_func(x, mu=0.01), x0, step_size=0.1, sigma=sigma, kappa=kappa, max_iter=2000, tol=1e-6)
    print("x_opt =", x_opt)
    print("f(x_opt)", obj_func(x_opt, mu=0.01))
    return x_opt

if __name__ == "__main__":
    run_example()