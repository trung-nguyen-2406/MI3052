import numpy as np

class MathExample1:
    def __init__(self):
        self.A = None
        self.b = None

    def calc_f(self, x):
        # Eq. (240)
        num = x[0]**2 + x[1]**2 + 3
        den = 1 + 2*x[0] + 8*x[1]
        return num / den

    def calc_grad_f(self, x):
        # Gradient of fractional function
        num = x[0]**2 + x[1]**2 + 3
        den = 1 + 2*x[0] + 8*x[1]
        
        grad_num = np.array([2*x[0], 2*x[1]])
        grad_den = np.array([2.0, 8.0])
        
        return (grad_num * den - num * grad_den) / (den**2)

    def calc_g(self, x):
        # Eq. (260)
        g1 = -x[0]**2 - 2*x[0]*x[1] + 4
        g2 = -x[0] 
        g3 = -x[1]
        return np.array([g1, g2, g3])

    def calc_grad_g(self, x):
        grad_g1 = np.array([-2*x[0] - 2*x[1], -2*x[0]])
        grad_g2 = np.array([-1.0, 0.0])
        grad_g3 = np.array([0.0, -1.0])
        return np.array([grad_g1, grad_g2, grad_g3])