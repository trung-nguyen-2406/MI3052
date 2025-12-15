import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import FrankWolfeSolver
from src.problems import Example2

if __name__ == "__main__":
    print("=== EXAMPLE 2: Nonsmooth Pseudoconvex Problem ===")
    
    # 1. Khởi tạo bài toán (4 biến)
    prob = Example2(n=4)
    
    # 2. Khởi tạo Solver
    solver = FrankWolfeSolver(prob, max_iter=200)
    
    # 3. Điểm khởi tạo x0
    # Cần thỏa mãn ràng buộc 2x1 + 4x2 + x3 = -1
    # Chọn x2=0, x1=-1, x3=1, x4=0 => -2 + 0 + 1 = -1 (Thỏa mãn Eq)
    # Check bất đẳng thức: g1 <= 10 (OK), g2 <= 1 (OK)
    x0 = [-1.0, 0.0, 1.0, 0.0]
    
    print(f"Start point: {x0}")
    
    # 4. Chạy thuật toán
    final_x = solver.solve(x0)
    
    print("\n" + "="*30)
    print("FINAL RESULT EXAMPLE 2:")
    print(f"Optimal x: {np.round(final_x, 4)}")
    print(f"Optimal f: {prob.objective_function(final_x):.6f}")
    
    print("-" * 30)
    print("Paper Reference (approx):")
    print("x* ~ [-1.0649, 0.4160, -0.5343, 0.0002]")
    print("f* ~ -3.0908")