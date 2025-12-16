import numpy as np

class FrankWolfeSolver:
    def __init__(self, problem, max_iter=200):
        """
        Solver nhận vào một đối tượng 'problem'.
        Đối tượng này phải có 3 hàm: objective_function, gradient, lmo.
        """
        self.problem = problem
        self.max_iter = max_iter

    def solve(self, x0, return_history=False):
        x = np.array(x0, dtype=float)
        
        # Tạo list để lưu lịch sử
        history = [x.copy()] 
        
        print(f"{'Iter':<5} | {'Cost f(x)':<15}")
        print("-" * 30)

        for k in range(self.max_iter):
            # 1. Gradient
            grad = self.problem.gradient(x)
            
            # 2. LMO
            s = self.problem.lmo(grad, x)
            
            # 3. Step size
            gamma = 2.0 / (k + 2.0)
            
            # 4. Update
            x_new = (1 - gamma) * x + gamma * s
            
            # Lưu lại giá trị x hiện tại vào lịch sử
            history.append(x_new.copy())
            
            # Log
            if k % 10 == 0 or k == self.max_iter - 1:
                val = self.problem.objective_function(x)
                print(f"{k:<5} | {val:.6f}")
                
            if np.linalg.norm(x_new - x) < 1e-7:
                break
                
            x = x_new
            
        # Trả về x cuối cùng và lịch sử history (nếu cần)
        if return_history:
            return x, np.array(history)
        else:
            return x