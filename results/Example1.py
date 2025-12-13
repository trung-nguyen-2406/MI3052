import numpy as np
from scipy.optimize import minimize
from manim import *

# Optional PyTorch for autograd-based NN optimizer. If unavailable,
# fallback to a NumPy finite-difference gradient descent.
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

def safe_math(expr, color=WHITE):
    """Return a MathTex (LaTeX) if available, otherwise Text fallback.
    This handles cases where MiKTeX/dvisvgm may not be fully configured.
    """
    # Prefer MathTex (LaTeX). If it fails, print a helpful warning and
    # fall back to Text so rendering doesn't completely abort.
    try:
        return MathTex(expr, color=color)
    except Exception as e:
        # Give a helpful message to the user about LaTeX/dvisvgm issues
        print(f"MathTex failed for '{expr}': {e}. Falling back to Text().\n"
              "If you want MathTex output, ensure a working LaTeX + dvisvgm toolchain (MiKTeX + dvisvgm>=2.4).")
        return Text(expr, color=color)

# --- PHẦN 1: THUẬT TOÁN GDA CÓ RÀNG BUỘC (Algorithm 1) ---

# --- 1.1. Hàm Mục Tiêu và Gradient (Không đổi) ---

def f(x):
    """Hàm mục tiêu: f(x) = (x1^2 + x2^2 + 3) / (1 + 2x1 + 8x2)"""
    x1, x2 = x
    numerator = x1**2 + x2**2 + 3
    denominator = 1 + 2*x1 + 8*x2
    if denominator <= 1e-9:
        return np.inf  
    return numerator / denominator

def grad_f(x):
    """Gradient của f(x)"""
    x1, x2 = x
    u = x1**2 + x2**2 + 3
    v = 1 + 2*x1 + 8*x2
    
    v_sq = v**2
    if v_sq < 1e-9:
        return np.array([np.inf, np.inf])

    df_dx1 = ((2*x1) * v - u * 2) / v_sq
    df_dx2 = ((2*x2) * v - u * 8) / v_sq
    
    return np.array([df_dx1, df_dx2])

# --- 1.2. Phép Chiếu P_C(y) sử dụng SciPy (Giải bài toán tối ưu phi lồi) ---

def projection_C(y):
    """
    P_C(y) = argmin ||z - y||^2 s.t. z1^2 + 2*z1*z2 >= 4, z1 >= 0, z2 >= 0
    Using trust-constr method like reference code
    """
    from scipy.optimize import BFGS, Bounds
    
    # Objective: minimize ||z - y||^2 (Euclidean distance squared)
    def objective_proj(z):
        return np.sum((z - y)**2)
    
    # Constraint: -z[0]^2 - 2*z[0]*z[1] + 4 <= 0  =>  z[0]^2 + 2*z[0]*z[1] - 4 >= 0
    def g1(z):
        return -z[0]**2 - 2*z[0]*z[1] + 4
    
    cons = {
        'type': 'ineq',
        'fun': lambda z: np.array([-g1(z)])
    }
    
    # Bounds: z >= 0
    bounds = Bounds([0, 0], [np.inf, np.inf])
    
    # Initial guess: random point (same as reference code)
    z0 = np.random.rand(2)
    
    # Solve using trust-constr with BFGS hessian (same as reference code)
    result = minimize(
        objective_proj, 
        z0,
        args=(),
        jac="2-point",  # Finite difference gradient
        hess=BFGS(),    # BFGS hessian approximation
        constraints=cons,
        method='trust-constr',
        options={'disp': False},
        bounds=bounds
    )
    
    if result.success:
        return result.x
    else:
        # Fallback
        print(f"Projection error at y={y}")
        return np.array([2.0, 0.0])

# --- 1.3. Thuật toán GDA (Algorithm 1) ---

def gda_solver(x0, lambda0, sigma, kappa, max_iter=100, tol=1e-8):
    """Thực thi Thuật toán Gradient Descent Adaptive (GDA)"""
    x = np.array(x0, dtype=float)
    lambda_k = float(lambda0)
    trajectory = []
    
    print(f"Starting GDA on non-convex set C. x0={x}")
    
    for k in range(max_iter):
        
        x_k = x.copy()
        
        # 1. Tính bước Gradient và Phép chiếu
        grad_xk = grad_f(x_k)
        if np.any(np.isinf(grad_xk)): break
            
        y = x_k - lambda_k * grad_xk
        x_new = projection_C(y)
        
        # 2. Adaptive Step Size
        dot_product = np.dot(grad_xk, x_k - x_new)
        RHS = f(x_k) - sigma * dot_product
        f_new = f(x_new)
        
        # Debug info
        print(f"  grad_xk: {grad_xk}")
        print(f"  x_k - x_new: {x_k - x_new}")
        print(f"  dot_product (should be > 0): {dot_product:.8f}")
        print(f"  sigma * dot_product: {sigma * dot_product:.8f}")
        
        if f_new <= RHS:
            lambda_next = lambda_k # Chấp nhận
            armijo_satisfied = "✓ Accept"
        else:
            lambda_next = kappa * lambda_k # Giảm lambda
            armijo_satisfied = "✗ Reduce"
        
        # Print detailed info at each iteration
        print(f"Iter {k}: λ={lambda_k:.6f} | f(x_k)={f(x_k):.6f}, f(x_new)={f_new:.6f}, RHS={RHS:.6f} | {armijo_satisfied} → λ_next={lambda_next:.6f}\n")

        # Lưu lại dữ liệu cho Manim
        trajectory.append({
            'k': k, 
            'x_start': x_k.copy(), 
            'y_mid': y.copy(), 
            'x_end': x_new.copy(), 
            'lambda_old': lambda_k,
            'lambda_new': lambda_next
        })

        # 3. Kiểm tra dừng (Step 2: If x^(k+1) = x^k then STOP)
        if np.linalg.norm(x_new - x_k) < tol:
            x = x_new
            lambda_k = lambda_next
            print(f"✅ GDA stopped at step k={k}. x*={x}, λ={lambda_k:.6f}")
            break
            
        x = x_new
        lambda_k = lambda_next
        
    return x, f(x), trajectory


# --- 1.4. Neural-network / gradient-based comparator (không thay đổi GDA) ---
def neural_network_solver(x0, lr=0.05, max_iter=50, penalty_weight=100.0):
    """
    So sánh bằng một phương pháp gradient bình thường:
    - Nếu PyTorch có sẵn: treat x as an nn.Parameter (or a tiny network output) and
      use SGD to minimize the objective + penalty for constraint violations.
    - Nếu không: fallback to simple finite-difference gradient descent on x.

    Trả về: final_x, f(final_x), trajectory (list of x at each iter)
    """
    x0 = np.array(x0, dtype=float)
    trajectory = []

    if TORCH_AVAILABLE:
        # Use a single parameter vector representing x
        x_param = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float32))
        optimizer = torch.optim.SGD([x_param], lr=lr)

        def f_torch(x_tensor):
            x1 = x_tensor[0]
            x2 = x_tensor[1]
            u = x1 * x1 + x2 * x2 + 3.0
            v = 1.0 + 2.0 * x1 + 8.0 * x2
            # Penalize non-positive denominator heavily
            if torch.abs(v) < 1e-6:
                return torch.tensor(1e6, dtype=torch.float32)
            return u / v

        for k in range(max_iter):
            optimizer.zero_grad()
            x_t = x_param
            loss_f = f_torch(x_t)

            # Soft penalty if constraint violated: g(x) = 4 - (x1^2 + 2 x1 x2) <= 0
            x1 = x_t[0]
            x2 = x_t[1]
            g_val = 4.0 - (x1 * x1 + 2.0 * x1 * x2)
            penalty = torch.relu(g_val) ** 2 * penalty_weight
            # Penalty for negative coordinates
            penalty = penalty + (torch.relu(-x1) ** 2 + torch.relu(-x2) ** 2) * (penalty_weight * 0.1)

            loss = loss_f + penalty
            loss.backward()
            optimizer.step()

            x_np = x_param.detach().cpu().numpy().astype(float)
            trajectory.append({'k': k, 'x': x_np.copy(), 'loss': float(loss.detach().cpu().numpy())})

        final_x = trajectory[-1]['x'] if len(trajectory) > 0 else x0
        return final_x, f(final_x), trajectory
    else:
        # Fallback: finite difference gradient descent on x (plain gradient method)
        x = x0.copy()
        eps = 1e-6
        for k in range(max_iter):
            # numerical gradient of L(x) = f(x) + penalty
            grad = np.zeros(2)
            base = f(x)
            for i in range(2):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                grad[i] = (f(xp) - f(xm)) / (2*eps)

            # add gradient from penalty: p(x) = relu(4 - (x1^2 + 2x1 x2))^2 * w
            g_val = 4.0 - (x[0]**2 + 2.0 * x[0] * x[1])
            if g_val > 0:
                # d/dx of (g^2) = 2 g * dg/dx
                dg_dx = np.array([-2.0*x[0] - 2.0*x[1], -2.0*x[0]])
                grad += 2.0 * g_val * dg_dx * penalty_weight

            # penalty for negative coordinates
            for i in range(2):
                if x[i] < 0:
                    grad[i] += 2.0 * (x[i]) * (penalty_weight * 0.1)

            # Update
            x = x - lr * grad
            trajectory.append({'k': k, 'x': x.copy(), 'loss': f(x)})

        final_x = trajectory[-1]['x'] if len(trajectory) > 0 else x0
        return final_x, f(final_x), trajectory

# --- 2. Chạy Solver để lấy Dữ liệu cho Manim ---
# Worse initialization far from optimal
x0_random = np.array([3.0, 0.1])  # Far from optimum
x0 = projection_C(x0_random)   # Project onto constraint set
print(f"Initial point (before projection): {x0_random}")
print(f"Projected initial point: {x0}")

# Start with small lambda (cannot be 0 or algorithm won't move)
lambda0 = 1.0      
# Use sigma = 0.1
sigma = 0.1         
kappa = 0.5           

final_x, final_f, trajectory_data = gda_solver(x0, lambda0, sigma, kappa, max_iter=40)

print(f"\nConverged solution: {final_x}, Objective value: {final_f}")

# --- Chạy phương pháp gradient bình thường (neural network / parameter-based) ---
nn_final_x, nn_final_f, nn_trajectory = neural_network_solver(x0, lr=0.05, max_iter=40, penalty_weight=100.0)
print(f"\nNN method result: {nn_final_x}, f: {nn_final_f} (torch_enabled={TORCH_AVAILABLE})")

# --- PHẦN 3: MINH HỌA QUÁ TRÌNH TÌM NGHIỆM BẰNG MANIM ---

class GDAProjectionAnimation(Scene):
    def construct(self):
        
        # --- 3.1. Thiết lập không gian và hàm số ---
        X_RANGE = [0, 4]
        Y_RANGE = [0, 4]
        
        axes = Axes(
            x_range=X_RANGE + [1],
            y_range=Y_RANGE + [1],
            x_length=8,
            y_length=8,
            axis_config={"include_numbers": False},
        ).to_edge(LEFT)
        # Defer axes appearance until after the title so elements appear sequentially
        # (we will create axes after writing the title below)

        # Ràng buộc C: x1^2 + 2*x1*x2 >= 4, x1 >= 0, x2 >= 0
        # Minh họa biên ràng buộc (Biên x1^2 + 2*x1*x2 = 4)
        
        def hyperbola_boundary(t):
            # t là x1
            # x2 = (4 - x1^2) / (2*x1)
            x1 = t
            x2 = (4 - x1**2) / (2*x1) if x1 > 0.01 else 4
            return axes.coords_to_point(x1, x2)

        # Vẽ biên (boundary) của miền ràng buộc C
        boundary_curve = ParametricFunction(
            hyperbola_boundary,
            t_range=[0.5, 4.0], # Bắt đầu từ x1 = 0.5 (4 - 0.25) / 1.0 = 3.75
            color=BLUE, 
            stroke_width=4
        )
        
        label_C = safe_math("C: x_1^2 + 2x_1x_2 \\geq 4", color=BLUE).next_to(boundary_curve, UP+RIGHT, buff=0.1)
        
        # Minh họa miền C (Miền nằm PHÍA NGOÀI và trên trục) 
        
        title = Text("GDA on Non-Convex Constraint Set").to_edge(UP)
        # Show title first, then create axes so elements appear sequentially
        self.play(Write(title), run_time=0.6)
        self.play(Create(axes), run_time=1)

        # Show the original objective function using MathTex (prefer LaTeX).
        # Shift it slightly left so it doesn't overlap the axes.
        f_display = safe_math("\\displaystyle f(x)=\\frac{x_1^2 + x_2^2 + 3}{1 + 2 x_1 + 8 x_2}", color=TEAL).scale(0.6)
        f_display.next_to(title, DOWN).shift(LEFT * 0.8)
        self.play(Write(f_display), run_time=1)

        # Axis labels for this single plot
        x_label = safe_math("x_1", color=WHITE).scale(0.6)
        y_label = safe_math("x_2", color=WHITE).scale(0.6)
        x_label.move_to(axes.coords_to_point(3.6, 0) + np.array([0.2, -0.25, 0]))
        y_label.move_to(axes.coords_to_point(0, 3.6) + np.array([-0.35, 0.15, 0]))
        self.play(Write(x_label), Write(y_label), run_time=0.6)

        self.play(Create(boundary_curve), Write(label_C), run_time=2)
        
        # --- 3.2. Minh họa từng bước GDA ---

        # Điểm khởi tạo
        x0_point = axes.coords_to_point(trajectory_data[0]['x_start'][0], trajectory_data[0]['x_start'][1])
        current_dot = Dot(x0_point, color=RED, radius=0.08)
        self.add(current_dot)

        # Khung thông tin (sử dụng MathTex nếu có, fallback là Text)
        info_text = VGroup(
            safe_math("k=", color=YELLOW_A),
            safe_math("x_k=", color=WHITE),
            safe_math("\\lambda_k=", color=GREEN),
            safe_math("f(x)=", color=TEAL),
            safe_math("\\kappa=", color=ORANGE)
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.6).to_edge(RIGHT).shift(2*UP)
        self.add(info_text)

        path_line = VGroup()
        # For less frequent rendering: accumulate segments and only render every `render_every` steps
        render_every = 2
        accumulated = []
        last_render_pos = trajectory_data[0]['x_start'].copy()

        for step in trajectory_data:
            k = step['k']
            x_k = step['x_start']
            y = step['y_mid']
            x_new = step['x_end']
            lambda_old = step['lambda_old']
            lambda_new = step.get('lambda_new', lambda_old)

            # convert to manim points
            p_xk = axes.coords_to_point(x_k[0], x_k[1])
            p_y = axes.coords_to_point(y[0], y[1])
            p_x_new = axes.coords_to_point(x_new[0], x_new[1])

            k_text = safe_math(f"k={k}", color=YELLOW_A).scale(0.6)
            xk_text = safe_math(f"x_k=({x_k[0]:.2f}, {x_k[1]:.2f})", color=WHITE).scale(0.6)
            lambda_text = safe_math(f"\\lambda_k={lambda_old:.3f} \\to {lambda_new:.3f}", color=GREEN).scale(0.6)
            fx_text = safe_math(f"f(x)={f(x_new):.4f}", color=TEAL).scale(0.6)
            kappa_text = safe_math(f"\\kappa={kappa:.3f}", color=ORANGE).scale(0.6)
            new_info_text = VGroup(k_text, xk_text, lambda_text, fx_text, kappa_text).arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT).shift(2*UP)

            # Create step visuals but only play them every `render_every` steps
            seg = Line(p_xk, p_x_new, color=YELLOW_A, stroke_width=2)
            accumulated.append(seg)

            # gradient/projection arrows only shown when rendering
            grad_line = Arrow(p_xk, p_y, buff=0.05, max_stroke_width_to_length_ratio=0.05, max_tip_length_to_length_ratio=0.15, color=GREEN)
            proj_line = None
            y_dot = None
            if np.linalg.norm(y - x_new) > 1e-4:
                proj_line = Arrow(p_y, p_x_new, buff=0.05, max_stroke_width_to_length_ratio=0.05, max_tip_length_to_length_ratio=0.15, color=RED)
                y_dot = Dot(p_y, color=ORANGE, radius=0.05)

            # If it's time to render, show accumulated path and arrows
            is_last_step = (k == len(trajectory_data) - 1)
            if k % render_every == 0 or is_last_step:
                # update info
                self.play(Transform(info_text, new_info_text), run_time=0.2)

                # show gradient/proj for this step
                if y_dot is not None:
                    self.play(Create(grad_line), FadeIn(y_dot), run_time=0.4)
                else:
                    self.play(Create(grad_line), run_time=0.4)
                if proj_line is not None:
                    self.play(Create(proj_line), run_time=0.4)
                    self.remove(grad_line, proj_line, y_dot)
                else:
                    self.remove(grad_line)

                # reveal accumulated segments as a group
                if len(accumulated) > 0:
                    self.play(*[Create(sg) for sg in accumulated], run_time=0.6)
                    path_line.add(*accumulated)
                    accumulated = []

                # move dot to current position with visible animation
                next_dot = Dot(p_x_new, color=RED, radius=0.08)
                self.play(Transform(current_dot, next_dot), run_time=0.5)
            else:
                # Not rendering this step: silently update current_dot position
                current_dot.move_to(p_x_new)
                path_line.add(seg)

        # --- 3.3. Kết luận ---
        
        final_x_manim = axes.coords_to_point(final_x[0], final_x[1])
        final_dot = Dot(final_x_manim, color=PURPLE, radius=0.15)
        
        self.play(
            Transform(current_dot, final_dot), 
            Flash(final_dot, color=PURPLE),
            FadeOut(info_text),
            run_time=2
        )
        
        final_text = VGroup(
            Text("Converged to local solution x*", color=PURPLE),
            Text(f"x* \approx ({final_x[0]:.4f}, {final_x[1]:.4f})", color=WHITE),
            Text(f"f(x*) \approx {final_f:.6f}", color=TEAL)
        ).arrange(DOWN).scale(0.7).to_edge(RIGHT).shift(1.5*DOWN)
        self.play(Write(final_text), run_time=2)
        
        self.wait(3)


class CombinedComparisonAnimation(Scene):
    """Animate GDA (left) and NN gradient method (right) side-by-side.
    This does not change the original GDA algorithm implementation; it merely
    visualizes both methods together for direct comparison.
    """
    def construct(self):
        # Shared ranges
        X_RANGE = [0, 4]
        Y_RANGE = [0, 4]

        # Left: GDA axes, Right: NN axes
        # Shift left axes further RIGHT to avoid hiding the x2 axis label
        axes_left = Axes(
            x_range=X_RANGE + [1], y_range=Y_RANGE + [1], x_length=6, y_length=6,
            axis_config={"include_numbers": False}
        ).to_edge(LEFT).shift(0.3*RIGHT)

        axes_right = Axes(
            x_range=X_RANGE + [1], y_range=Y_RANGE + [1], x_length=6, y_length=6,
            axis_config={"include_numbers": False}
        ).to_edge(RIGHT).shift(0.7*RIGHT)

        title = Text("GDA (Left)  vs  Gradient/NN (Right)").scale(0.7).to_edge(UP)
        # Appear axes and title sequentially
        self.play(Create(axes_left), run_time=0.7)
        self.play(Create(axes_right), run_time=0.7)
        self.play(Write(title), run_time=0.6)

        # Show the original objective function (centered under the title)
        f_display_center = safe_math("\\displaystyle f(x)=\\frac{x_1^2 + x_2^2 + 3}{1 + 2 x_1 + 8 x_2}", color=TEAL).scale(0.6)
        f_display_center.next_to(title, DOWN).shift(LEFT * 0.8)
        self.play(Write(f_display_center), run_time=1)

        # Show the constraint set definition below the objective
        c_display = safe_math("C: \\, x_1^2 + 2x_1 x_2 \\geq 4, \\, x_1 \\geq 0, \\, x_2 \\geq 0", color=BLUE).scale(0.5)
        c_display.next_to(f_display_center, DOWN, buff=0.2)
        self.play(Write(c_display), run_time=0.8)

        # Axis labels for left plot
        x_label_left = safe_math("x_1", color=WHITE).scale(0.5)
        y_label_left = safe_math("x_2", color=WHITE).scale(0.5)
        x_label_left.move_to(axes_left.coords_to_point(3.2, 0) + np.array([0.18, -0.2, 0]))
        y_label_left.move_to(axes_left.coords_to_point(0, 3.2) + np.array([-0.28, 0.12, 0]))
        self.play(Write(x_label_left), Write(y_label_left), run_time=0.3)

        # Axis labels for right plot
        x_label_right = safe_math("x_1", color=WHITE).scale(0.5)
        y_label_right = safe_math("x_2", color=WHITE).scale(0.5)
        x_label_right.move_to(axes_right.coords_to_point(3.2, 0) + np.array([0.18, -0.2, 0]))
        y_label_right.move_to(axes_right.coords_to_point(0, 3.2) + np.array([-0.28, 0.12, 0]))
        self.play(Write(x_label_right), Write(y_label_right), run_time=0.3)

        # Draw boundary curve for C on both plots (clearer than dots)
        def hyperbola_boundary_left(t):
            x1 = t
            x2 = (4 - x1**2) / (2*x1) if x1 > 0.01 else 4
            return axes_left.coords_to_point(x1, x2)

        def hyperbola_boundary_right(t):
            x1 = t
            x2 = (4 - x1**2) / (2*x1) if x1 > 0.01 else 4
            return axes_right.coords_to_point(x1, x2)

        boundary_left = ParametricFunction(hyperbola_boundary_left, t_range=[0.5, 4.0], color=BLUE, stroke_width=3)
        boundary_right = ParametricFunction(hyperbola_boundary_right, t_range=[0.5, 4.0], color=BLUE, stroke_width=3)

        # optional light fill: sample dots for visibility (kept low-density)
        feasible_dots_left = VGroup()
        feasible_dots_right = VGroup()
        xs = np.linspace(0.2, 4.0, 30)
        ys = np.linspace(0.2, 4.0, 30)
        for xi in xs:
            for yi in ys:
                if xi*xi + 2*xi*yi >= 4 - 1e-8:
                    p_left = axes_left.coords_to_point(xi, yi)
                    p_right = axes_right.coords_to_point(xi, yi)
                    d1 = Dot(p_left, color=BLUE, radius=0.02).set_opacity(0.25)
                    d2 = Dot(p_right, color=BLUE, radius=0.02).set_opacity(0.25)
                    feasible_dots_left.add(d1)
                    feasible_dots_right.add(d2)

        # Draw boundaries and then fade in feasible region points
        self.play(Create(boundary_left), Create(boundary_right), run_time=1)
        self.play(FadeIn(feasible_dots_left), FadeIn(feasible_dots_right), run_time=0.8)

        # Prepare initial markers
        gda_start = trajectory_data[0]['x_start'] if len(trajectory_data) > 0 else np.array(x0)
        gda_dot = Dot(axes_left.coords_to_point(gda_start[0], gda_start[1]), color=RED, radius=0.07)
        nn_start = nn_trajectory[0]['x'] if len(nn_trajectory) > 0 else np.array(x0)
        nn_dot = Dot(axes_right.coords_to_point(nn_start[0], nn_start[1]), color=ORANGE, radius=0.07)

        self.add(gda_dot, nn_dot)

        # info boxes for both methods
        # Position GDA info BELOW the constraint set definition
        gda_info = VGroup(
            safe_math("GDA: k=0", color=YELLOW_A),
            safe_math("x=(--, --)", color=WHITE),
            safe_math("\\lambda=--", color=GREEN),
            safe_math("\\kappa=%0.3f" % (kappa,), color=ORANGE)
        ).arrange(DOWN).scale(0.5).next_to(c_display, DOWN, buff=0.2)

        nn_info = VGroup(
            safe_math("NN: k=0", color=YELLOW_A),
            safe_math("x=(--, --)", color=WHITE),
            safe_math("loss=--", color=GREEN)
        ).arrange(DOWN).scale(0.5).to_edge(RIGHT).shift(1*UP)

        self.add(gda_info, nn_info)

        # Draw trajectories step-by-step in sync for up to the longer length but render every `render_every` steps
        n_gda = len(trajectory_data)
        n_nn = len(nn_trajectory)
        n_steps = max(n_gda, n_nn)

        gda_path = VGroup()
        nn_path = VGroup()

        render_every = 2
        accumulated_gda = []
        accumulated_nn = []

        for t in range(n_steps):
            # GDA step
            if t < n_gda:
                s = trajectory_data[t]
                p_old = axes_left.coords_to_point(s['x_start'][0], s['x_start'][1])
                p_new = axes_left.coords_to_point(s['x_end'][0], s['x_end'][1])
                seg = Line(p_old, p_new, color=YELLOW_A, stroke_width=2)
                accumulated_gda.append(seg)
            else:
                seg = None

            # NN step
            if t < n_nn:
                s2 = nn_trajectory[t]
                p_old2 = axes_right.coords_to_point(s2['x'][0], s2['x'][1])
                # Next position (if available) otherwise keep
                if t+1 < n_nn:
                    p_new2 = axes_right.coords_to_point(nn_trajectory[t+1]['x'][0], nn_trajectory[t+1]['x'][1])
                else:
                    p_new2 = p_old2
                seg2 = Line(p_old2, p_new2, color=GREEN, stroke_width=2)
                accumulated_nn.append(seg2)
            else:
                seg2 = None

            # Render only every `render_every` steps: update infos, reveal accumulated segments, move markers visibly
            is_last_step = (t == n_steps - 1)
            if t % render_every == 0 or is_last_step:
                # update infos
                if t < n_gda:
                    g = trajectory_data[t]
                    gda_info_new = VGroup(
                        safe_math(f"GDA: k={g['k']}", color=YELLOW_A),
                        safe_math(f"x=({g['x_end'][0]:.2f}, {g['x_end'][1]:.2f})", color=WHITE),
                        safe_math(f"\\lambda={g['lambda_old']:.3f}", color=GREEN),
                        safe_math(f"f={f(np.array(g['x_end'])):.4f}", color=TEAL),
                        safe_math(f"\\kappa={kappa:.3f}", color=ORANGE)
                    ).arrange(DOWN).scale(0.5).next_to(c_display, DOWN, buff=0.2)
                    self.play(Transform(gda_info, gda_info_new), run_time=0.2)

                if t < n_nn:
                    nval = nn_trajectory[t]
                    nn_info_new = VGroup(
                        safe_math(f"NN: k={nval['k']}", color=YELLOW_A),
                        safe_math(f"x=({nval['x'][0]:.2f}, {nval['x'][1]:.2f})", color=WHITE),
                        safe_math(f"loss={nval.get('loss', 0):.4f}", color=GREEN),
                        safe_math(f"f={f(np.array(nval['x'])):.4f}", color=TEAL)
                    ).arrange(DOWN).scale(0.5).to_edge(RIGHT).shift(1*UP)
                    self.play(Transform(nn_info, nn_info_new), run_time=0.2)

                if len(accumulated_gda) > 0:
                    self.play(*[Create(sg) for sg in accumulated_gda], run_time=0.6)
                    gda_path.add(*accumulated_gda)
                    accumulated_gda = []

                if len(accumulated_nn) > 0:
                    self.play(*[Create(sg) for sg in accumulated_nn], run_time=0.6)
                    nn_path.add(*accumulated_nn)
                    accumulated_nn = []

                # move dots visibly
                if t < n_gda:
                    self.play(Transform(gda_dot, Dot(axes_left.coords_to_point(trajectory_data[t]['x_end'][0], trajectory_data[t]['x_end'][1]), color=RED, radius=0.07)), run_time=0.4)
                if t < n_nn:
                    self.play(Transform(nn_dot, Dot(axes_right.coords_to_point(nn_trajectory[t]['x'][0], nn_trajectory[t]['x'][1]), color=ORANGE, radius=0.07)), run_time=0.4)

        # Highlight final points
        final_gda = Dot(axes_left.coords_to_point(final_x[0], final_x[1]), color=PURPLE, radius=0.12)
        final_nn = Dot(axes_right.coords_to_point(nn_final_x[0], nn_final_x[1]), color=PURPLE, radius=0.12)
        self.play(Transform(gda_dot, final_gda), Transform(nn_dot, final_nn), run_time=1.5)

        # Display final objective values for both methods
        final_stats = VGroup(
            safe_math(f"GDA: x* = ({final_x[0]:.4f}, {final_x[1]:.4f}), f={final_f:.6f}", color=WHITE),
            safe_math(f"NN: x* = ({nn_final_x[0]:.4f}, {nn_final_x[1]:.4f}), f={nn_final_f:.6f}", color=WHITE)
        ).arrange(DOWN).to_edge(DOWN)
        self.play(Write(final_stats))

        self.wait(3)