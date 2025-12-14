import numpy as np
import matplotlib.pyplot as plt
import time
from src.problem.math_example_1 import f, grad_f, g, grad_g

# Nghiệm tối ưu lý thuyết (Theo bài báo Example 1 / 5.2):
TRUE_OPTIMAL_X = np.array([0.8907, 1.7957])
TRUE_OPTIMAL_VAL = 0.4094

def solve_with_rnn(f, grad_f, g, grad_g, start_point, max_iter=2000, step_size=0.01):
    x = np.array(start_point, dtype=float)

    history = {
        'trajectory': [x.copy()],
        'obj_values': [f(x)],
        'g_values': [g(x)],
        'timestamps': [0]
    }

    start_time = time.time()

    for i in range(max_iter):
        # 1. Logic chuyển mạch
        val_g = g(x)
        if val_g > 0:
            c = 0.0      # Vi phạm -> Tắt mục tiêu
            psi = 1.0    # Bật sửa lỗi
        else:
            c = 1.0      # An toàn -> Tắt sửa lỗi
            psi = 0.0    # Bật mục tiêu

        # 2. Tính vận tốc (Phương trình động lực học)
        # dx/dt = -c*grad_f - psi*grad_g
        dxdt = -c * grad_f(x) - psi * grad_g(x)

        # 3. Cập nhật (Euler)
        x = x + dxdt * step_size
        x = np.maximum(x, 0.0001) # Giữ dương (x >= 0)

        # 4. Lưu dữ liệu
        curr_time = time.time() - start_time
        history['trajectory'].append(x.copy())
        history['obj_values'].append(f(x))
        history['g_values'].append(g(x))
        history['timestamps'].append(curr_time)

    return x, history

# ==========================================
# PHẦN 3: BỘ CÔNG CỤ ĐÁNH GIÁ
# ==========================================

def evaluate_performance(algorithm_name, final_x, history):
    trajectory = np.array(history['trajectory'])
    obj_vals = np.array(history['obj_values'])
    g_vals = np.array(history['g_values'])

    final_obj_val = obj_vals[-1]
    obj_error = abs(final_obj_val - TRUE_OPTIMAL_VAL)
    pos_error = np.linalg.norm(final_x - TRUE_OPTIMAL_X)
    final_violation = max(0, g_vals[-1])

    diffs = np.diff(trajectory, axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=1))

    # Tính thời gian hội tụ (khi sai số < 1e-3)
    epsilon = 1e-3
    converge_idx = np.where(np.abs(obj_vals - TRUE_OPTIMAL_VAL) < epsilon)[0]
    if len(converge_idx) > 0:
        time_to_converge = history['timestamps'][converge_idx[0]]
        iter_to_converge = converge_idx[0]
    else:
        time_to_converge = history['timestamps'][-1]
        iter_to_converge = len(obj_vals)

    print(f"\n--- BẢNG ĐÁNH GIÁ: {algorithm_name} ---")
    print(f"1. Điểm kết thúc (Final X):     [{final_x[0]:.4f}, {final_x[1]:.4f}]")
    print(f"2. Sai số mục tiêu (Cost Err):  {obj_error:.6f}")
    print(f"3. Sai số vị trí (Dist Err):    {pos_error:.6f}")
    print(f"4. Vi phạm ràng buộc (Final):   {final_violation:.6f}")
    print(f"5. Độ dài đường đi (Total Path):{path_length:.4f}")
    print(f"6. Thời gian hội tụ (s):        {time_to_converge:.4f}s \t(Tại iter {iter_to_converge})")

    return {
        'obj_error': obj_error,
        'time_to_converge': time_to_converge
    }


if __name__ == "__main__":
    start_pos = [0.1, 0.1]

    print(f"Đang chạy RNN từ điểm {start_pos}...")
    final_x, hist = solve_with_rnn(f, grad_f, g, grad_g, start_pos, max_iter=3000, step_size=0.005)

    metrics = evaluate_performance("RNN (Paper 1)", final_x, hist)

    # Vẽ đồ thị
    plt.figure(figsize=(12, 4))

    # Hình 1: Quỹ đạo
    traj = np.array(hist['trajectory'])
    plt.subplot(1, 2, 1)
    plt.plot(traj[:,0], traj[:,1], label='RNN Path')
    plt.scatter(TRUE_OPTIMAL_X[0], TRUE_OPTIMAL_X[1], c='r', marker='*', s=150, label='Optimal')
    plt.scatter(start_pos[0], start_pos[1], c='g', label='Start')

    # Vẽ đường ràng buộc g(x)=0 => 4 - x1^2 - 2x1x2 = 0 => x2 = (4 - x1^2)/(2x1)
    x1_vals = np.linspace(0.1, 2.5, 100)
    x2_vals = (4 - x1_vals**2) / (2*x1_vals)
    plt.plot(x1_vals, x2_vals, 'k--', label='Biên g(x)=0')

    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.title('Quỹ đạo di chuyển (x1, x2)')
    plt.legend()
    plt.grid()

    # Hình 2: Hàm mục tiêu
    plt.subplot(1, 2, 2)
    plt.plot(hist['obj_values'])
    plt.axhline(y=TRUE_OPTIMAL_VAL, color='r', linestyle='--')
    plt.title('Hội tụ hàm mục tiêu f(x)')
    plt.xlabel('Iterations')
    plt.grid()

    plt.tight_layout()
    plt.show()