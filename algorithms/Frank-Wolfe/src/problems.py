import numpy as np
from scipy.optimize import minimize

# ==========================================
# EXAMPLE 1
# ==========================================
class Example1:
    def __init__(self, n=2):
        self.n = n

    def objective_function(self, x):
        x1, x2 = x[0], x[1]
        numerator = x1**2 + x2**2 + 3
        denominator = 1 + 2*x1 + 8*x2
        return numerator / denominator

    def gradient(self, x):
        x1, x2 = x[0], x[1]
        num = x1**2 + x2**2 + 3
        denom = 1 + 2*x1 + 8*x2
        df_dx1 = (2*x1 * denom - num * 2) / (denom**2)
        df_dx2 = (2*x2 * denom - num * 8) / (denom**2)
        return np.array([df_dx1, df_dx2])

    def lmo(self, grad, current_x):
        # Frank-Wolfe LMO
        # Ràng buộc: -x1^2 - 2x1x2 <= -4
        cons = ({'type': 'ineq', 'fun': lambda s: s[0]**2 + 2*s[0]*s[1] - 4})
        # Chỉnh bound cho đồ thị dễ nhìn
        bnds = [(0, 5)] * self.n
        # start point = current_x giúp solver chạy nhanh hơn
        res = minimize(lambda s: np.dot(grad, s), current_x, bounds=bnds, constraints=cons, method='SLSQP')
        return res.x

# ==========================================
# EXAMPLE 2
# ==========================================
class Example2:
    def __init__(self, n=4):
        self.n = n

    def objective_function(self, x):
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        term1 = np.exp(np.abs(x2 - 3)) - 30
        term2 = x1**2 + x3**2 + 2*x4**2 + 4
        return term1 / term2

    def gradient(self, x):
        eps = 1e-8
        grad = np.zeros(4)
        for i in range(4):
            x_plus = x.copy(); x_minus = x.copy()
            x_plus[i] += eps; x_minus[i] -= eps
            grad[i] = (self.objective_function(x_plus) - self.objective_function(x_minus)) / (2 * eps)
        return grad

    def lmo(self, grad, current_x):
        cons = (
            {'type': 'ineq', 'fun': lambda y: 10 - ((y[0] + y[2])**2 + 2*y[3]**2)}, # g1 <= 10
            {'type': 'ineq', 'fun': lambda y: 1 - (y[1] - 1)**2},                   # g2 <= 1
            {'type': 'eq',   'fun': lambda y: 2*y[0] + 4*y[1] + y[2] + 1}           # eq = 0
        )
        bnds = [(-50, 50)] * 4
        res = minimize(lambda s: np.dot(grad, s), current_x, bounds=bnds, constraints=cons, method='SLSQP')
        return res.x

# ==========================================
# EXAMPLE 3
# ==========================================
class Example3:
    def __init__(self, n=10):
        self.n = n
        # COPY Y HỆT KHỐI INIT CỦA GDA ĐỂ KHỚP RANDOM SEED
        np.random.seed(42)
        self.a = np.abs(np.random.randn(n))
        self.e = np.array([i for i in range(1, n + 1)])
        self.beta = 0.741271
        self.alpha = 3 * (self.beta ** 1.5) * np.sqrt(n + 1)

    def objective_function(self, x):
        term1 = np.dot(self.a, x)
        term2 = self.alpha * np.dot(x, x)
        xTx = np.dot(x, x)
        sqrt_term = np.sqrt(1 + self.beta * xTx)
        term3 = self.beta / sqrt_term * np.dot(self.e, x)
        return term1 + term2 + term3

    def gradient(self, x):
        grad = self.a.copy()
        grad += 2 * self.alpha * x
        xTx = np.dot(x, x)
        sqrt_term = np.sqrt(1 + self.beta * xTx)
        term3_part1 = self.beta / sqrt_term
        grad += term3_part1 * self.e
        d_sqrt = -self.beta**2 * xTx / (2 * (sqrt_term**3))
        grad += d_sqrt * (np.dot(self.e, x)) * (2 * x)
        return grad

    def lmo(self, grad, current_x):
        def constraint_prod(s):
            return np.sum(np.log(s + 1e-10))

        cons = ({'type': 'ineq', 'fun': constraint_prod})
        bnds = [(1e-5, 1000)] * self.n
        
        res = minimize(lambda s: np.dot(grad, s), current_x, bounds=bnds, constraints=cons, method='SLSQP')
        return res.x

# ==========================================
# EXAMPLE 4
# ==========================================
class Example4:
    def __init__(self, n=10):
        self.n = n
        np.random.seed(42)
        self.rho = abs(np.random.uniform(low = 0, high = 100, size = self.n))
        self.A = np.ones(n)
        for i in range(n):
            if i >= n // 2:
                self.A[i] = 3.0
        self.b = 16.0

    def objective_function(self, x):
        sum_term = np.sum((x / self.rho) ** 2)
        return -np.exp(-sum_term)

    def gradient(self, x):
        sum_term = np.sum((x / self.rho) ** 2)
        exp_term = np.exp(-sum_term)
        grad = 2 * exp_term * x / (self.rho ** 2)
        return grad

    def lmo(self, grad, current_x):
        # Constraints: Ax = b
        linear_cons = {'type': 'eq', 'fun': lambda s: np.dot(self.A, s) - self.b}
        
        # Constraints: g_i(x) <= 0
        ineq_cons = []
        num_groups = self.n // 10
        for i in range(num_groups):
            def make_constraint(idx):
                start = 10 * idx
                end = min(start + 10, self.n)
                # Chuyển về dạng >= 0 cho scipy: 20 - sum(...) >= 0
                return lambda s: 20 - np.sum(s[start:end]**2)
            ineq_cons.append({'type': 'ineq', 'fun': make_constraint(i)})

        all_cons = [linear_cons] + ineq_cons
        bnds = [(-100, 100)] * self.n
        
        res = minimize(lambda s: np.dot(grad, s), current_x, bounds=bnds, constraints=all_cons, method='SLSQP')
        return res.x
    
"""
Copy logic hàm mục tiêu và đạo hàm từ bên GDA. (tks)
Thay thế projection bằng lmo.
"""