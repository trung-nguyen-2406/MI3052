import numpy as np

class MathExample4:
    def __init__(self, n=10):
        self.n = n
        
        # Matrix A construction: Half 1s, Half 3s
        self.row_A = np.zeros(n)
        mid = n // 2
        self.row_A[:mid] = 1.0
        self.row_A[mid:] = 3.0
        
        self.A = self.row_A.reshape(1, -1)
        self.b = np.array([16.0])
        
        # Scaling parameter
        self.e_sq = np.ones(n) 

    def calc_f(self, x):
        # Eq. (310)
        sum_sq = np.sum((x**2) / self.e_sq)
        return -np.exp(-sum_sq)

    def calc_grad_f(self, x):
        sum_sq = np.sum((x**2) / self.e_sq)
        exp_term = np.exp(-sum_sq)
        
        grad_u = 2 * x / self.e_sq
        return exp_term * grad_u

    def calc_g(self, x):
        # Eq. (321): Block constraints
        constraints = []
        num_blocks = self.n // 10
        if num_blocks == 0: return np.array([-1.0])
        
        for i in range(num_blocks):
            start = i * 10
            end = start + 10
            block = x[start:end]
            val = np.sum(block**2) - 20
            constraints.append(val)
            
        return np.array(constraints)

    def calc_grad_g(self, x):
        num_blocks = self.n // 10
        if num_blocks == 0: return np.zeros((1, self.n))
        
        grads = np.zeros((num_blocks, self.n))
        for i in range(num_blocks):
            start = i * 10
            end = start + 10
            grads[i, start:end] = 2 * x[start:end]
            
        return grads