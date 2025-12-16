import numpy as np
from typing import Tuple
from scipy.optimize import minimize
import numpy as np
from typing import Callable, Optional, Dict, Any
import time

def create_example_4(n : int) : 
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
    
    np.random.seed(42)
    rho = abs(np.random.uniform(low = 0, high = 100, size = n))
    # for i in range(n):
    #     if i >= n // 2:
    #         rho[i] = 3.0
    
    # Create constraint matrix A (1×n)
    A = np.ones(n)
    for i in range(n):
        if i >= n // 2:
            A[i] = 3.0
    
    b = 16.0
    
    def obj_func(x: np.ndarray) -> float:
        """Objective function f(x) = -exp(-Σ(xi^2/ρi^2))"""
        sum_term = np.sum((x / rho) ** 2)
        return -np.exp(-sum_term)
    
    def grad_func(x: np.ndarray) -> np.ndarray:
        """Gradient of f(x)"""
        sum_term = np.sum((x / rho) ** 2)
        exp_term = np.exp(-sum_term)
        
        # d/dx_i[-exp(-Σ(x_j^2/ρ_j^2))] = exp(-Σ(x_j^2/ρ_j^2)) * 2*x_i/ρ_i^2
        grad = 2 * exp_term * x / (rho ** 2)
        
        return grad
    
    def proj_func(x: np.ndarray) -> np.ndarray:
        """Project onto feasible set C = {Ax = b, g(x) ≤ 0}"""
        
        def constraint_eq(y):
            """Ax - b = 0"""
            return np.dot(A, y) - b
        
        def constraint_ineq(y):
            """g_i(x) = sum of 10 consecutive x_j² - 20 ≤ 0"""
            constraints = []
            for i in range(n // 10):
                start_idx = i * 10
                end_idx = min(start_idx + 10, n)
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
    np.random.seed(42)
    x0 = np.random.randn(n) * 2.0
    return obj_func, grad_func, proj_func, x0


class GDAOptimizer:
    """
    Gradient Descent Adaptive (GDA) Algorithm for nonconvex and quasiconvex optimization
    
    Algorithm 1 (GDA):
    Step 0: Choose x^0 ∈ C, λ_0 ∈ (0, +∞), σ, κ ∈ (0, 1). Set k = 0.
    Step 1: Given x^k and λ_k, compute x^{k+1} and λ_{k+1} as:
            x^{k+1} = P_C(x^k - λ_k ∇f(x^k))
            If f(x^{k+1}) ≤ f(x^k) - σ⟨∇f(x^k), x^k - x^{k+1}⟩ then set λ_{k+1} = λ_k else set λ_{k+1} = κλ_k
    Step 2: Update k := k + 1. If x^{k+1} = x^k then STOP else go to Step 1.
    """
    
    def __init__(self, 
                 obj_func: Callable,
                 grad_func: Callable,
                 x0: np.ndarray,
                 proj_func: Optional[Callable] = None,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 lambda_0: float = 1.0,
                 sigma: float = 0.1,
                 kappa: float = 0.5,
                 verbose: bool = False):
        """
        Initialize GDA optimizer
        
        Parameters:
        -----------
        func : Callable
            Objective function f(x)
        grad_func : Callable
            Gradient function ∇f(x)
        x0 : np.ndarray
            Initial point in feasible set C
        proj_func : Callable, optional
            Projection operator P_C(x). If None, identity projection (unconstrained)
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for convergence
        lambda_0 : float
            Initial step size λ_0
        sigma : float
            Sufficient decrease parameter σ ∈ (0, 1)
        kappa : float
            Step size reduction factor κ ∈ (0, 1)
        verbose : bool
            Print progress information
        """
        self.obj_func = obj_func
        self.grad_func = grad_func
        self.x0 = np.copy(x0)
        self.proj_func = proj_func if proj_func is not None else lambda x: x
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_0 = lambda_0
        self.sigma = sigma
        self.kappa = kappa
        self.verbose = verbose
        
        self.history = {
            'x': [],
            'f': [],
            'grad_norm': [],
            'step_size': [],
            'iterations': 0
        }
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run GDA optimization algorithm
        
        Returns:
        --------
        Dict with optimization results
        """
        x = np.copy(self.x0)
        lambda_k = self.lambda_0
        
        start_time = time.time()
        
        for k in range(self.max_iter):
            # Compute gradient at current point
            grad = self.grad_func(x)
            grad_norm = np.linalg.norm(grad)
            
            # Current function value
            f_val = self.obj_func(x)
            
            # Store history
            self.history['x'].append(np.copy(x))
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            self.history['step_size'].append(lambda_k)
            
            if self.verbose and (k % 10 == 0 or k < 10):
                print(f"Iter {k}: f(x) = {f_val:.6f}, ||grad|| = {grad_norm:.6e}, lambda = {lambda_k:.6e}")
            
            # Check convergence: if gradient is small enough
            # if grad_norm < self.tol:
            #     if self.verbose:
            #         print(f"Converged at iteration {k}")
            #     break
            
            # Step 1: Compute x^{k+1} = P_C(x^k - λ_k ∇f(x^k))
            x_new = self.proj_func(x - lambda_k * grad)
            f_new = self.obj_func(x_new)
            
            # Compute inner product ⟨∇f(x^k), x^k - x^{k+1}⟩
            direction = x - x_new
            inner_product = np.dot(grad, direction)
            
            # Check sufficient decrease condition:
            # f(x^{k+1}) ≤ f(x^k) - σ⟨∇f(x^k), x^k - x^{k+1}⟩
            if f_new <= f_val - self.sigma * inner_product:
                # Condition satisfied: keep step size
                lambda_k = lambda_k
                # if self.verbose and k < 20:
                #     print(f"  -> Decrease satisfied, keep λ = {lambda_k:.6e}")
            else:
                # Condition not satisfied: reduce step size
                lambda_k = self.kappa * lambda_k
                # if self.verbose and k < 20:
                #     print(f"  -> Decrease failed, reduce λ = {lambda_k:.6e}")
            
            # Update x
            
            # Step 2: Check if converged (x^{k+1} = x^k)
            if np.allclose(x, x_new, atol=self.tol, rtol=1e-5):
                if self.verbose:
                    print(f"Stationary point reached at iteration {k+1}")
                break
            x = x_new
        
        elapsed_time = time.time() - start_time
        self.history['iterations'] = k + 1
        
        result = {
            'x': x,
            'f': self.obj_func(x),
            'iterations': k + 1,
            'time': elapsed_time,
            'history': self.history,
            'converged': grad_norm < self.tol
        }
        
        return result


import sys 

sys.path.insert(0, '..')

def run_example_4():
    """Run Example 4 individually"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: GAUSSIAN EXPONENTIAL PROBLEM")
    print("="*80)
    
    # Create example with different dimensions
    n_values = [10, 20, 50, 100, 300, 400, 600]
    
    print(f"""
Problem Definition:
    minimize f(x) = -exp(-Σ(x_i^2/σ_i^2))
    
    where:
        σ_i = 1.0 for i = 1, ..., n/2
        σ_i = 3.0 for i = n/2+1, ..., n
    
Comparing Algorithm GDA with Algorithm RNN (Recurrent Neural Network)
from Liu et al. (2022) for different dimensions n.
    """)
    
    print("=" * 80)
    print("COMPUTATIONAL RESULTS FOR EXAMPLE 4")
    print("=" * 80)
    print(f"{'n':<10} {'-ln(-f(x*))':<15} {'GDA Iter':<12} {'GDA Time(s)':<12}")
    print("-" * 50)
    
    results_gda = []
    
    for n in n_values:
        print(f"{n:<10}", end=" ", flush=True)
        
        # Create example
        obj_func, grad_func, proj_func, x0 = create_example_4(n)
        
        # Fixed lambda_0 that works well
        lambda_0 = 0.5
        
        # Run GDA with fixed 10 iterations as per paper
        gda = GDAOptimizer(
            obj_func= obj_func,
            grad_func=grad_func,
            x0=x0,
            proj_func=proj_func,
            max_iter=10,  # Fixed at 10 as per Table 2
            tol=1e-12,
            lambda_0=lambda_0,
            sigma=0.1,
            kappa=0.5,
            verbose=False
        )
        
        gda_result = gda.optimize()
        results_gda.append(gda_result)
        
        # Display -ln(-f(x*))
        f_val = gda_result['f']
        if f_val < 0:
            display_val = -np.log(-f_val)
        else:
            display_val = f_val
        
        print(f"{display_val:<15.4f} {gda_result['iterations']:<12} {gda_result['time']:<12.4f}")
    
    print()
    
if __name__ == "__main__":
    run_example_4()

