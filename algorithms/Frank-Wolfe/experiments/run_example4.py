import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import FrankWolfeSolver
from src.problems import Example4

if __name__ == "__main__":
    # n phải chia hết cho 10
    n_dim = 20
    print(f"=== EXAMPLE 4: Gaussian Exponential (n={n_dim}) ===")
    
    # 1. Khởi tạo bài toán
    prob = Example4(n=n_dim)
    
    # 2. Solver (Cần tăng max_iter)
    solver = FrankWolfeSolver(prob, max_iter=500)
    
    # 3. Điểm khởi tạo x0
    # Ràng buộc Ax = 16. 
    # Với n=20, A có 10 số 1 và 10 số 3. Tổng trọng số = 10*1 + 10*3 = 40.
    # Để Ax = 16 -> Trung bình mỗi x khoảng 16/40 = 0.4
    # Chọn x0 = [0.4, 0.4, ..., 0.4]
    x0 = [0.4] * n_dim
    
    # 4. Chạy
    final_x = solver.solve(x0)
    
    print("\n" + "="*30)
    print(f"FINAL RESULT EXAMPLE 4 (n={n_dim}):")
    print(f"Optimal x (first 5): {np.round(final_x[:5], 4)} ...")
    print(f"Optimal f: {prob.objective_function(final_x):.6f}")
    
    # print("-" * 30)
    # print("Paper Reference:")
    # print("Giá trị f tối ưu thường là số âm gần 0 (vì f = -exp(...))")
    # print("Ví dụ n=20 bảng trong bài báo ra khoảng -ln(-f) = 2.56 => f = -e^(-2.56) ~ -0.077")