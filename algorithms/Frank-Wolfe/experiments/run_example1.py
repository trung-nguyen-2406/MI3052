import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import FrankWolfeSolver
from src.problems import Example1

if __name__ == "__main__":
    print("=== EXAMPLE 1: Simple Nonconvex Problem ===")
    
    # 1. Khởi tạo bài toán
    prob = Example1(n=2)
    
    # 2. Khởi tạo Solver
    solver = FrankWolfeSolver(prob, max_iter=200)
    
    # 3. Điểm khởi tạo x0
    # Ràng buộc: -x1^2 - 2x1x2 <= -4 (tức là x1^2 + 2x1x2 >= 4)
    # Chọn x = [2, 2] => 4 + 8 = 12 >= 4 (Thỏa mãn)
    x0 = [2.0, 2.0]
    
    print(f"Start point: {x0}")
    
    # 4. Chạy thuật toán
    final_x = solver.solve(x0)
    
    print("\n" + "="*30)
    print("FINAL RESULT EXAMPLE 1:")
    print(f"Optimal x: {np.round(final_x, 4)}")
    print(f"Optimal f: {prob.objective_function(final_x):.6f}")
    
    print("-" * 30)
    print("Paper Reference:")
    print("x* approx: [0.8922, 1.7957]")
    print("f* approx: 0.4094")