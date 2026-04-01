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


# --- 1.4. Projected Gradient Descent (GD) solver ---

def projected_gradient_descent(x0, lambda0, max_iter=100, tol=1e-8):
    """
    Projected Gradient Descent with fixed step size
    
    Algorithm:
    1. y = x - lambda * grad_f(x)
    2. x_new = P_C(y)  (projection onto constraint set)
    3. Check convergence
    
    Args:
        x0: initial point
        lambda0: step size (learning rate)
        max_iter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        final_x, final_f, trajectory
    """
    x = np.array(x0, dtype=float)
    lambda_k = float(lambda0)
    trajectory = []
    
    print(f"Starting Projected GD. x0={x}, lambda={lambda_k}")
    
    for k in range(max_iter):
        x_k = x.copy()
        
        # 1. Gradient step
        grad_xk = grad_f(x_k)
        if np.any(np.isinf(grad_xk)):
            break
        
        # 2. Gradient descent step
        y = x_k - lambda_k * grad_xk
        
        # 3. Project onto constraint set
        x_new = projection_C(y)
        
        f_val = f(x_new)
        
        # Save trajectory data
        trajectory.append({
            'k': k,
            'x_start': x_k.copy(),
            'y_mid': y.copy(),
            'x_end': x_new.copy(),
            'lambda': lambda_k,
            'f': f_val
        })
        
        print(f"Iter {k}: λ={lambda_k:.6f} | x=({x_new[0]:.4f}, {x_new[1]:.4f}), f={f_val:.6f}")
        
        # 4. Check convergence
        if np.linalg.norm(x_new - x_k) < tol:
            x = x_new
            print(f"✅ GD converged at step k={k}. x*={x}, λ={lambda_k:.6f}")
            break
        
        x = x_new
    
    return x, f(x), trajectory

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

# --- Chạy phương pháp Projected Gradient Descent ---
gd_lambda = 0.1  # Fixed step size for GD
gd_final_x, gd_final_f, gd_trajectory = projected_gradient_descent(x0, gd_lambda, max_iter=40)
print(f"\nProjected GD result: {gd_final_x}, f: {gd_final_f}")

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
    """Animate GDA (left) and Projected Gradient Descent (right) side-by-side.
    This does not change the original GDA algorithm implementation; it merely
    visualizes both methods together for direct comparison.
    """
    def construct(self):
        # Shared ranges
        X_RANGE = [0, 4]
        Y_RANGE = [0, 4]

        # Left: GDA axes, Right: GD axes
        # Shift left axes further RIGHT to avoid hiding the x2 axis label
        axes_left = Axes(
            x_range=X_RANGE + [1], y_range=Y_RANGE + [1], x_length=6, y_length=6,
            axis_config={"include_numbers": False}
        ).to_edge(LEFT).shift(0.3*RIGHT)

        axes_right = Axes(
            x_range=X_RANGE + [1], y_range=Y_RANGE + [1], x_length=6, y_length=6,
            axis_config={"include_numbers": False}
        ).to_edge(RIGHT).shift(0.7*RIGHT)

        title = Text("GDA (Left)  vs  GD (Right)").scale(0.7).to_edge(UP)
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
        gd_start = gd_trajectory[0]['x_start'] if len(gd_trajectory) > 0 else np.array(x0)
        gd_dot = Dot(axes_right.coords_to_point(gd_start[0], gd_start[1]), color=ORANGE, radius=0.07)

        self.add(gda_dot, gd_dot)

        # info boxes for both methods
        # Position GDA info BELOW the constraint set definition
        gda_info = VGroup(
            safe_math("GDA: k=0", color=YELLOW_A),
            safe_math("x=(--, --)", color=WHITE),
            safe_math("\\lambda=--", color=GREEN),
            safe_math("\\kappa=%0.3f" % (kappa,), color=ORANGE)
        ).arrange(DOWN).scale(0.5).next_to(c_display, DOWN, buff=0.2)

        gd_info = VGroup(
            safe_math("GD: k=0", color=YELLOW_A),
            safe_math("x=(--, --)", color=WHITE),
            safe_math("\\lambda=--", color=GREEN),
            safe_math("f=--", color=TEAL)
        ).arrange(DOWN).scale(0.5).to_edge(RIGHT).shift(1*UP)

        self.add(gda_info, gd_info)

        # Draw trajectories step-by-step in sync for up to the longer length but render every `render_every` steps
        n_gda = len(trajectory_data)
        n_gd = len(gd_trajectory)
        n_steps = max(n_gda, n_gd)

        gda_path = VGroup()
        gd_path = VGroup()

        render_every = 2
        accumulated_gda = []
        accumulated_gd = []

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

            # GD step
            if t < n_gd:
                s2 = gd_trajectory[t]
                p_old2 = axes_right.coords_to_point(s2['x_start'][0], s2['x_start'][1])
                p_new2 = axes_right.coords_to_point(s2['x_end'][0], s2['x_end'][1])
                seg2 = Line(p_old2, p_new2, color=GREEN, stroke_width=2)
                accumulated_gd.append(seg2)
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

                if t < n_gd:
                    gval = gd_trajectory[t]
                    gd_info_new = VGroup(
                        safe_math(f"GD: k={gval['k']}", color=YELLOW_A),
                        safe_math(f"x=({gval['x_end'][0]:.2f}, {gval['x_end'][1]:.2f})", color=WHITE),
                        safe_math(f"\\lambda={gval['lambda']:.3f}", color=GREEN),
                        safe_math(f"f={gval['f']:.4f}", color=TEAL)
                    ).arrange(DOWN).scale(0.5).to_edge(RIGHT).shift(1*UP)
                    self.play(Transform(gd_info, gd_info_new), run_time=0.2)

                if len(accumulated_gda) > 0:
                    self.play(*[Create(sg) for sg in accumulated_gda], run_time=0.6)
                    gda_path.add(*accumulated_gda)
                    accumulated_gda = []

                if len(accumulated_gd) > 0:
                    self.play(*[Create(sg) for sg in accumulated_gd], run_time=0.6)
                    gd_path.add(*accumulated_gd)
                    accumulated_gd = []

                # move dots visibly
                if t < n_gda:
                    self.play(Transform(gda_dot, Dot(axes_left.coords_to_point(trajectory_data[t]['x_end'][0], trajectory_data[t]['x_end'][1]), color=RED, radius=0.07)), run_time=0.4)
                if t < n_gd:
                    self.play(Transform(gd_dot, Dot(axes_right.coords_to_point(gd_trajectory[t]['x_end'][0], gd_trajectory[t]['x_end'][1]), color=ORANGE, radius=0.07)), run_time=0.4)

        # Highlight final points
        final_gda = Dot(axes_left.coords_to_point(final_x[0], final_x[1]), color=PURPLE, radius=0.12)
        final_gd = Dot(axes_right.coords_to_point(gd_final_x[0], gd_final_x[1]), color=PURPLE, radius=0.12)
        self.play(Transform(gda_dot, final_gda), Transform(gd_dot, final_gd), run_time=1.5)

        # Display final objective values for both methods
        final_stats = VGroup(
            safe_math(f"GDA: x* = ({final_x[0]:.4f}, {final_x[1]:.4f}), f={final_f:.6f}", color=WHITE),
            safe_math(f"GD: x = ({gd_final_x[0]:.4f}, {gd_final_x[1]:.4f}), f={gd_final_f:.6f}", color=WHITE)
        ).arrange(DOWN).to_edge(DOWN)
        self.play(Write(final_stats))

        self.wait(3)