import numpy as np

class MathExample2:
    def __init__(self):
        # Equality constraints: 2x1 + 4x2 + x3 = -1
        self.A = np.array([[2.0, 4.0, 1.0, 0.0]])
        self.b = np.array([-1.0])

    def calc_f(self, x):
        # Eq. (266)
        term_abs = np.abs(x[1] - 3)
        num = np.exp(term_abs) - 30
        den = x[0]**2 + x[1]**2 + 2*x[2]**2 + 4
        return num / den

    def calc_grad_f(self, x):
        term_abs = np.abs(x[1] - 3)
        num = np.exp(term_abs) - 30
        den = x[0]**2 + x[1]**2 + 2*x[2]**2 + 4
        
        # Subgradient for |x2 - 3|
        sign_x2 = np.sign(x[1] - 3)
        if sign_x2 == 0: sign_x2 = 0.0
        
        grad_num = np.zeros(4)
        grad_num[1] = np.exp(term_abs) * sign_x2
        grad_den = np.array([2*x[0], 2*x[1], 4*x[2], 0.0])
        
        return (grad_num * den - num * grad_den) / (den**2)

    def calc_g(self, x):
        # Eq. (268-269)
        g1 = (x[0] + x[2])**3 + 2*x[3]**2 - 10
        g2 = (x[1] - 1)**2 - 1
        return np.array([g1, g2])

    def calc_grad_g(self, x):
        dg1 = np.array([3*(x[0] + x[2])**2, 0.0, 3*(x[0] + x[2])**2, 4*x[3]])
        dg2 = np.array([0.0, 2*(x[1] - 1), 0.0, 0.0])
        return np.array([dg1, dg2])