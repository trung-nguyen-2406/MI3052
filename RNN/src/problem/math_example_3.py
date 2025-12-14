import numpy as np

class MathExample3:
    def __init__(self, n=10):
        self.n = n
        self.A = None
        self.b = None
        
        # Parameters from paper
        self.beta = 0.741271
        self.alpha = 3 * (self.beta**1.5) * np.sqrt(n + 1)
        self.a_vec = np.ones(n)

    def calc_f(self, x):
        # Eq. (301)
        xtx = np.dot(x, x)
        term1 = np.dot(self.a_vec, x)
        term2 = self.alpha * xtx
        
        sum_x = np.sum(x)
        denom = np.sqrt(1 + self.beta * xtx)
        term3 = (self.beta / denom) * sum_x
        
        return term1 + term2 + term3

    def calc_grad_f(self, x):
        xtx = np.dot(x, x)
        sum_x = np.sum(x)
        root_term = np.sqrt(1 + self.beta * xtx)
        
        grad_t1_t2 = self.a_vec + 2 * self.alpha * x
        
        term3_const = self.beta / root_term
        grad_sum = np.ones(self.n)
        
        # Chain rule for term 3
        grad_inv_root = -0.5 * (1 + self.beta * xtx)**(-1.5) * (2 * self.beta * x)
        grad_term3 = (grad_sum * term3_const) + (sum_x * self.beta * grad_inv_root)
        
        return grad_t1_t2 + grad_term3

    def calc_g(self, x):
        # Eq. (303): 1 <= Prod(x_i)
        prod_val = np.prod(x)
        return np.array([1.0 - prod_val])

    def calc_grad_g(self, x):
        prod_val = np.prod(x)
        # Avoid division by zero
        grad = -1.0 * (prod_val / (x + 1e-10))
        return np.array([grad])