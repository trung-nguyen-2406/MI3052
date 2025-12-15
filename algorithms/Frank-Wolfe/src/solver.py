import numpy as np

class FrankWolfeSolver:
    def __init__(self, problem, max_iter=200):
        """
        Solver nhận vào một đối tượng 'problem'.
        Đối tượng này phải có 3 hàm: objective_function, gradient, lmo.
        """
        self.problem = problem
        self.max_iter = max_iter

    def solve(self, x0):
        x = np.array(x0, dtype=float)
        
        print(f"{'Iter':<5} | {'Cost f(x)':<15}")
        print("-" * 30)

        for k in range(self.max_iter):
            # 1. Tính Gradient (gọi từ problem)
            grad = self.problem.gradient(x)
            
            # 2. LMO (gọi từ problem)
            s = self.problem.lmo(grad, x)
            
            # 3. Step size chuẩn
            gamma = 2.0 / (k + 2.0)
            
            # 4. Update
            x_new = (1 - gamma) * x + gamma * s
            
            # Log
            if k % 10 == 0 or k == self.max_iter - 1:
                val = self.problem.objective_function(x)
                print(f"{k:<5} | {val:.6f}")
                
            # Check convergence simple
            if np.linalg.norm(x_new - x) < 1e-7:
                break
                
            x = x_new
            
        return x