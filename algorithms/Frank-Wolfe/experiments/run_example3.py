import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import FrankWolfeSolver
from src.problems import Example3

if __name__ == "__main__":
    # Thay đổi n tại đây (10, 20, 50, 100...)
    n_dim = 20
    print(f"=== EXAMPLE 3: Large-scale Problem (n={n_dim}) ===")
    
    # 1. Khởi tạo bài toán
    prob = Example3(n=n_dim)
    
    # 2. Solver (Bài này hàm lồi, FW chạy khá tốt)
    solver = FrankWolfeSolver(prob, max_iter=200)
    
    # 3. Điểm khởi tạo x0
    # Ràng buộc: x > 0 và Tích các phần tử >= 1
    # Chọn an toàn: Toàn bộ là số 2.0 (Vì 2^20 chắc chắn > 1)
    x0 = [2.0] * n_dim
    
    # 4. Chạy
    final_x = solver.solve(x0)
    
    print("\n" + "="*30)
    print(f"FINAL RESULT EXAMPLE 3 (n={n_dim}):")
    # Chỉ in 5 phần tử đầu tiên cho đỡ rối
    print(f"Optimal x (first 5): {np.round(final_x[:5], 4)} ...")
    print(f"Optimal f: {prob.objective_function(final_x):.6f}")
    
    # print("-" * 30)
    # print("Note: Kết quả sẽ khác nhau tùy thuộc vào Random Seed của vector a")
    # print("Nhưng code này đã fix seed=42 giống GDA nên kết quả sẽ so sánh được.")