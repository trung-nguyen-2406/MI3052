"""
Example implementations of the 4 test problems from the paper
"""

import numpy as np
from typing import Tuple
from scipy.optimize import minimize


class Example1:
    """
    Example 1: Simple nonconvex problem
    minimize f(x) = (x1^2 + x2^2 + 3) / (1 + 2*x1 + 8*x2)
    subject to x ∈ C where C = {x = (x1, x2) | -x1^2 - 2*x1*x2 ≤ -4, x1, x2 ≥ 0}
    """
    
    name = "Example 1: Nonconvex Problem"
    
    @staticmethod
    def objective(x: np.ndarray) -> float:
        """Objective function f(x)"""
        x1, x2 = x[0], x[1]
        numerator = x1**2 + x2**2 + 3
        denominator = 1 + 2*x1 + 8*x2
        return numerator / denominator
    
    @staticmethod
    def gradient(x: np.ndarray) -> np.ndarray:
        """Gradient of f(x)"""
        x1, x2 = x[0], x[1]
        num = x1**2 + x2**2 + 3
        denom = 1 + 2*x1 + 8*x2
        
        # Derivative with respect to x1
        df_dx1 = (2*x1 * denom - num * 2) / (denom**2)
        
        # Derivative with respect to x2
        df_dx2 = (2*x2 * denom - num * 8) / (denom**2)
        
        return np.array([df_dx1, df_dx2])
    
    @staticmethod
    def projection(x: np.ndarray) -> np.ndarray:
        """Project onto feasible set C"""
        # C = {x : -x1^2 - 2*x1*x2 ≤ -4, x1 ≥ 0, x2 ≥ 0}
        # Equivalently: x1^2 + 2*x1*x2 ≥ 4
        # For simplicity, project x to non-negative and satisfy constraint
        x_proj = np.maximum(x, 0)
        
        # Check constraint: -x1^2 - 2*x1*x2 ≤ -4
        # i.e., x1^2 + 2*x1*x2 ≥ 4
        x1, x2 = x_proj[0], x_proj[1]
        if x1**2 + 2*x1*x2 < 4:
            # Adjust to satisfy constraint
            if x1 > 0:
                x2 = (4 - x1**2) / (2*x1)
            x_proj = np.array([x1, x2])
        
        return x_proj
    
    @staticmethod
    def feasible_point() -> np.ndarray:
        """Return a feasible initial point"""
        # For Example 1, we need x1^2 - 2*x1*x2 ≤ -4
        # A feasible point: x = (2, 2)
        return np.array([2.0, 2.0])


class Example2:
    """
    Example 2: Nonsmooth pseudoconvex optimization with nonconvex inequality constraints
    minimize f(x) = (e^|x2-3| - 30) / (x1^2 + x3^2 + 2*x4^2 + 4)
    subject to:
        g1(x) = (x1 + x3)^2 + 2*x4^2 ≤ 10
        g2(x) = (x2 - 1)^2 ≤ 1
        2*x1 + 4*x2 + x3 = -1
    """
    
    name = "Example 2: Nonsmooth Pseudoconvex Problem"
    
    @staticmethod
    def objective(x: np.ndarray) -> float:
        """Objective function f(x) = (e^|x2-3| - 30) / (x1^2 + x3^2 + 2*x4^2 + 4)"""
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        term1 = np.exp(np.abs(x2 - 3)) - 30
        term2 = x1**2 + x3**2 + 2*x4**2 + 4
        return term1 / term2
    
    @staticmethod
    def gradient(x: np.ndarray) -> np.ndarray:
        """Gradient of f(x) using numerical differentiation"""
        eps = 1e-8
        grad = np.zeros(4)
        
        for i in range(4):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (Example2.objective(x_plus) - Example2.objective(x_minus)) / (2 * eps)
        
        return grad
    
    @staticmethod
    def projection(x: np.ndarray) -> np.ndarray:
        """Project onto feasible set C using numerical optimization"""
        
        def constraint_g2(y):
            """g2(x) = (x2 - 1)^2 - 1 ≤ 0"""
            return (y[1] - 1)**2 - 1
        
        def constraint_g1(y):
            """g1(x) = (x1 + x3)^2 + 2*x4^2 - 10 ≤ 0"""
            return (y[0] + y[2])**2 + 2*y[3]**2 - 10
        
        def constraint_eq(y):
            """2*x1 + 4*x2 + x3 + 1 = 0"""
            return 2*y[0] + 4*y[1] + y[2] + 1
        
        def objective(y):
            """Minimize distance to original point x"""
            return np.sum((y - x)**2)
        
        # Constraints for scipy.optimize
        constraints = [
            {'type': 'ineq', 'fun': constraint_g1},
            {'type': 'ineq', 'fun': constraint_g2},
            {'type': 'eq', 'fun': constraint_eq}
        ]
        
        # Initial guess
        x0 = x.copy()
        
        # Solve projection problem
        result = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                         options={'ftol': 1e-9, 'maxiter': 1000})
        
        if result.success:
            return result.x
        else:
            # Fallback: simple feasibility fix
            x_proj = x.copy()
            x_proj[1] = np.clip(x_proj[1], 0, 2)  # g2
            x_proj[2] = -1 - 2*x_proj[0] - 4*x_proj[1]  # equality
            
            # Check g1 and scale if needed
            g1_val = (x_proj[0] + x_proj[2])**2 + 2*x_proj[3]**2
            if g1_val > 10:
                scale = np.sqrt(10 / (g1_val + 1e-10))
                x_proj[0] *= scale
                x_proj[3] *= scale
                x_proj[2] = -1 - 2*x_proj[0] - 4*x_proj[1]
            
            return x_proj
    
    @staticmethod
    def feasible_point() -> np.ndarray:
        """Return a feasible initial point"""
        # x = (-1.0, 0.0, 0.0, 0.0)
        return np.array([-1.0, 0.0, 0.0, 0.0])


class Example3:
    """
    Example 3: Large-scale problem
    Let e := (1, ..., 1) ∈ ℝⁿ be a vector. α > 0 and β > 0 be constants satisfying 
    parameter condition 2α > 3β^(3/2)/√n.
    
    f(x) = aᵀx + αxᵀx + (β/√(1 + βxᵀx)) eᵀx
    
    Constraint: C = {x ∈ ℝⁿ₊₊ : 1 ≤ x₁...xₙ}
    
    Parameters: β = 0.741271, α = 3β^(3/2)/√(n+1)
    """
    
    name = "Example 3: Large-scale Problem"
    
    def __init__(self, n: int = 10):
        self.n = n
        # Use fixed seed for reproducibility - a must be in ℝⁿ₊₊ (all positive)
        np.random.seed(42)
        self.a = np.abs(np.random.randn(n))  # Ensure a > 0
        self.e = np.array([i for i in range(1, n + 1)])
        # print(self.e)
        # print(self.a)
        self.beta = 0.741271
        self.alpha = 3 * (self.beta ** 1.5) * np.sqrt(n + 1)
    
    def objective(self, x: np.ndarray) -> float:
        """Objective function f(x)"""
        term1 = np.dot(self.a, x)
        term2 = self.alpha * np.dot(x, x)
        
        xTx = np.dot(x, x)
        sqrt_term = np.sqrt(1 + self.beta * xTx)
        term3 = self.beta / sqrt_term * np.dot(self.e, x)
        
        return term1 + term2 + term3
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of f(x)"""
        grad = self.a.copy()
        grad += 2 * self.alpha * x
        
        xTx = np.dot(x, x)
        sqrt_term = np.sqrt(1 + self.beta * xTx)
        
        # Derivative of β/√(1 + β*x^T*x) * e^T*x
        term3_part1 = self.beta / sqrt_term
        grad += term3_part1 * self.e
        
        # Chain rule for the square root part
        d_sqrt = -self.beta**2 * xTx / (2 * (sqrt_term**3))
        grad += d_sqrt * (np.dot(self.e, x)) * (2 * x)
        
        return grad
    
    # def projection(self, x: np.ndarray) -> np.ndarray:
    #     """Project onto feasible set C = {x : xi ≥ 1 for all i}"""
    #     return np.maximum(x, 1.0)

    # def projection(sefl, v : np.ndarray):
    #     from scipy.optimize import root_scalar
    #     """
    #     Tính hình chiếu Euclide của v lên tập C = {x > 0 : prod(x) >= 1}.
        
    #     Parameters:
    #     v (np.array): Vector đầu vào (có thể là kết quả của bước Gradient Descent).
        
    #     Returns:
    #     x (np.array): Hình chiếu của v lên tập C.
    #     """
    #     if np.all(v > 0) and np.prod(v) >= 1:
    #         return v
        
    #     def constraint_func(mu):
    #         # Tính x dựa trên mu
    #         # mu phải dương. Sử dụng broadcasting của numpy.
    #         delta = np.sqrt(v**2 + 4 * mu)
    #         x_mu = (v + delta) / 2
    #         # Điều kiện biên là tích x_i = 1, hay tổng ln(x_i) = 0
    #         return np.sum(np.log(x_mu))

    #     upper_bound = 1.0
    #     while constraint_func(upper_bound) < 0:
    #         upper_bound *= 10
            
    #     sol = root_scalar(constraint_func, bracket=[0, upper_bound], method='brentq')
    #     mu_opt = sol.root
    #     x_projected = (v + np.sqrt(v**2 + 4 * mu_opt)) / 2
        
    #     return x_projected

    # def projection(self, z : np.ndarray, eps=1e-12):
    #     z_pos = np.maximum(z, eps)
    #     log_prod = np.sum(np.log(z_pos))

    #     if log_prod >= 0:  # log(1) = 0
    #         return z_pos

    #     n = z_pos.size
    #     t = np.exp(-log_prod / n)
    #     return t * z_pos

    def projection(self, x: np.ndarray) -> np.ndarray:
        from scipy.optimize import root_scalar
        """
        Phép chiếu vector x lên tập C = {z in R++^n : prod(z) >= 1}.
        Bài toán: min ||z - x||^2 s.t. sum(log(z_i)) >= 0
        """
        # 1. Kiểm tra nếu x đã thuộc C (Tích x_i >= 1 hay Tổng log(x_i) >= 0)
        # Sử dụng log để tránh tràn số (overflow) với n lớn
        if np.sum(np.log(x)) >= 0:
            return x

        # 2. Nếu không, điểm chiếu nằm trên biên (prod(z) = 1)
        # Chúng ta cần tìm nhân tử Lagrange mu > 0 sao cho:
        # sum( log( (x_i + sqrt(x_i^2 + 4*mu)) / 2 ) ) = 0
        
        def equation(mu):
            # Công thức nghiệm từ điều kiện KKT: z_i = (x_i + sqrt(x_i^2 + 4*mu)) / 2
            # mu là biến số cần tìm
            z = (x + np.sqrt(x**2 + 4 * mu)) / 2.0
            return np.sum(np.log(z))

        # 3. Giải phương trình tìm mu (mu phải dương)
        # Hàm log tăng dần theo mu, nên ta có thể dùng phương pháp Brent hoặc Bisect
        # Bracket [0, 1e5] là khoảng tìm kiếm, có thể cần chỉnh nếu mu quá lớn
        try:
            sol = root_scalar(equation, bracket=[0, 1e6], method='brentq')
            mu_opt = sol.root
        except ValueError:
            # Trường hợp hiếm: nếu không tìm thấy nghiệm trong khoảng, mở rộng khoảng
            sol = root_scalar(equation, bracket=[0, 1e12], method='brentq')
            mu_opt = sol.root

        # 4. Tính vector kết quả z dựa trên mu tối ưu
        z_projected = (x + np.sqrt(x**2 + 4 * mu_opt)) / 2.0
        
        return z_projected    
    
    def feasible_point(self) -> np.ndarray:
        """Return a feasible initial point (all ones)"""
        np.random.seed(42)
        x = np.random.rand(1, self.n).tolist()[0]
        return x
        # return np.random.uniform(low = -100, high = 100, size = self.n)


class Example4:
    """
    Example 4: Gaussian exponential problem (from Liu et al. 2022)
    minimize f(x) = -exp(-Σ(xi^2/ρi^2))
    subject to Ax = b, g(x) ≤ 0
    
    where:
        ρi = 1 for i = 1,...,n/2
        ρi = 3 for i = n/2+1,...,n
        A = [a₁, a₂, ..., aₙ] ∈ ℝ^(1×n) with aᵢ = 1 for i ≤ n/2, aᵢ = 3 for i > n/2
        b = 16
        gᵢ(x) = x²₁₀₍ᵢ₋₁₎₊₁ + ... + x²₁₀₍ᵢ₋₁₎₊₁₀ - 20 ≤ 0 for i = 1,...,n/10
    """
    
    name = "Example 4: Gaussian Exponential Problem"
    
    def __init__(self, n: int = 10):
        self.n = n
        # Create ρ vector
        np.random.seed(42)
        self.rho = abs(np.random.uniform(low = 0, high = 100, size = self.n))
        # for i in range(n):
        #     if i >= n // 2:
        #         self.rho[i] = 3.0
        
        # Create constraint matrix A (1×n)
        self.A = np.ones(n)
        for i in range(n):
            if i >= n // 2:
                self.A[i] = 3.0
        
        self.b = 16.0
        # print(self.rho)
    
    def objective(self, x: np.ndarray) -> float:
        """Objective function f(x) = -exp(-Σ(xi^2/ρi^2))"""
        sum_term = np.sum((x / self.rho) ** 2)
        return -np.exp(-sum_term)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of f(x)"""
        sum_term = np.sum((x / self.rho) ** 2)
        exp_term = np.exp(-sum_term)
        
        # d/dx_i[-exp(-Σ(x_j^2/ρ_j^2))] = exp(-Σ(x_j^2/ρ_j^2)) * 2*x_i/ρ_i^2
        grad = 2 * exp_term * x / (self.rho ** 2)
        
        return grad
    
    def projection(self, x: np.ndarray) -> np.ndarray:
        """Project onto feasible set C = {Ax = b, g(x) ≤ 0}"""
        from scipy.optimize import minimize
        
        def constraint_eq(y):
            """Ax - b = 0"""
            return np.dot(self.A, y) - self.b
        
        def constraint_ineq(y):
            """g_i(x) = sum of 10 consecutive x_j² - 20 ≤ 0"""
            constraints = []
            for i in range(self.n // 10):
                start_idx = i * 10
                end_idx = min(start_idx + 10, self.n)
                g_i = np.sum(y[start_idx:end_idx] ** 2) - 20
                constraints.append(-g_i)  # Convert to ≤ 0 form
            return np.array(constraints) if constraints else np.array([0])
        
        def objective(y):
            """Minimize distance to original point x"""
            return np.sum((y - x) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': constraint_eq},
            {'type': 'ineq', 'fun': constraint_ineq}
        ]
        
        # Solve projection problem
        result = minimize(objective, x.copy(), method='SLSQP', 
                         constraints=constraints,
                         options={'ftol': 1e-10, 'maxiter': 100})
        
        return result.x
        # if result.success:
        # else:
        #     # Fallback: project to equality only
        #     Ax = np.dot(self.A, x)
        #     A_norm_sq = np.dot(self.A, self.A)
        #     lambda_val = (self.b - Ax) / A_norm_sq
        #     return x + lambda_val * self.A
        
        # return x_proj
    
    def feasible_point(self) -> np.ndarray:
        """Return a feasible initial point"""
        # Initialize with random point and project to feasible set
        np.random.seed(42)
        x = np.random.randn(self.n) * 2.0
        
        # Project to satisfy Ax = b
        # x = self.projection(x)
        
        return x


# Function to create examples with different dimensions
def create_example(example_num: int, n: int = 10):
    """
    Create an example problem
    
    Parameters:
    -----------
    example_num : int
        Example number (1-4)
    n : int
        Problem dimension (for examples 3 and 4)
    
    Returns:
    --------
    Tuple of (objective_func, gradient_func, projection_func, x0, name)
    """
    if example_num == 1:
        example = Example1()
        return (example.objective, example.gradient, example.projection, example.feasible_point(), example.name)
    
    elif example_num == 2:
        example = Example2()
        return (example.objective, example.gradient, example.projection, example.feasible_point(), example.name)
    
    elif example_num == 3:
        example = Example3(n)
        return (example.objective, example.gradient, example.projection, example.feasible_point(), example.name)
    
    elif example_num == 4:
        example = Example4(n)
        return (example.objective, example.gradient, example.projection, example.feasible_point(), example.name)
    
    else:
        raise ValueError("Example number must be 1-4")
