import sys
import os
import numpy as np
import matplotlib.pyplot as plt 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import FrankWolfeSolver
from src.problems import Example1

if __name__ == "__main__":
    print("=== EXAMPLE 1: Simple Nonconvex Problem ===")
    
    # 1. Khởi tạo bài toán
    prob = Example1(n=2)
    solver = FrankWolfeSolver(prob, max_iter=1000)
    
    # 2. Điểm khởi tạo 
    x0 = [2.0, 2.0]
    
    # 3. Chạy thuật toán (Nhận về cả history)
    final_x, history = solver.solve(x0, return_history=True)
    
    print("\n" + "="*30)
    print("FINAL RESULT EXAMPLE 1:")
    print(f"Optimal x: {np.round(final_x, 4)}")
    print(f"Optimal f: {prob.objective_function(final_x):.6f}")
    
    print("-" * 30)
    print("Paper Reference:")
    print("x* approx: [0.8922, 1.7957]")
    print("f* approx: 0.4094")

    # ==========================================
    # PHẦN VẼ BIỂU ĐỒ
    # ==========================================
    
    # Tách lịch sử ra thành 2 mảng riêng: x1 và x2
    # history là ma trận [số vòng lặp, 2]
    x1_values = history[:, 0]
    x2_values = history[:, 1]
    iterations = range(len(history))

    # Vẽ
    plt.figure(figsize=(10, 6))
    
    # Đường x1 (Màu đỏ)
    plt.plot(iterations, x1_values, color='red', label=r'$x_1(t)$', linewidth=2)
    
    # Đường x2 (Màu xanh lá)
    plt.plot(iterations, x2_values, color='green', label=r'$x_2(t)$', linewidth=2)
    
    # Deco
    plt.title('Computation Results: Frank-Wolfe Algorithm (Example 1)', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Value x(t)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7) # Kẻ ô lưới nét đứt
    
    # Hiển thị
    plt.tight_layout()
    plt.show()