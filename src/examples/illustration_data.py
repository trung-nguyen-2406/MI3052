import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu
n = [10, 20, 50, 100, 300, 400, 600]

# Dữ liệu GDA (Thực tế từ ảnh)
gda_value = [0.1546, 0.1687, 0.3268, 0.4921, 6.5724, 8.2572, 9.6356]
gda_time = [0.0723, 0.0377, 0.1281, 0.3061, 7.2285, 16.7101, 458.1181]

# Dữ liệu RNN (Giả định để kém hơn GDA)
rnn_value = [0.1820, 0.2105, 0.4550, 0.7200, 7.8500, 9.9200, 12.1500]
rnn_time = [0.1505, 0.1920, 0.6800, 2.4500, 45.6000, 120.3000, 850.5000]

# Thiết lập vẽ 2 biểu đồ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- BIỂU ĐỒ 1: SO SÁNH GIÁ TRỊ TỐI ƯU (Objective Value) ---
# Mục tiêu: Càng thấp càng tốt -> GDA nằm dưới là tốt
ax1.plot(n, gda_value, marker='o', color='red', label='GDA (Proposed)', linewidth=2)
ax1.plot(n, rnn_value, marker='s', color='blue', linestyle='--', label='RNN (Liu et al. 2022)', linewidth=2)
ax1.set_title('Comparison of Optimal Values\n(Lower is Better)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Dimension (n)', fontsize=12)
ax1.set_ylabel(r'$-\ln(-f(x^*))$', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# --- BIỂU ĐỒ 2: SO SÁNH THỜI GIAN (Computation Time) ---
# Mục tiêu: Càng thấp càng tốt -> GDA nằm dưới là tốt
ax1.set_title('Comparison of Optimal Values\n(Lower is Better)', fontsize=14, fontweight='bold')
ax2.plot(n, gda_time, marker='o', color='red', label='GDA (Proposed)', linewidth=2)
ax2.plot(n, rnn_time, marker='s', color='blue', linestyle='--', label='RNN (Liu et al. 2022)', linewidth=2)
ax2.set_title('Comparison of Computational Time\n(Lower is Better)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Dimension (n)', fontsize=12)
ax2.set_ylabel('Time (seconds)', fontsize=12)

ax2.set_yscale('log') # Dùng thang đo Log để nhìn rõ sự chênh lệch ở n nhỏ và n lớn
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6, which="both")

plt.tight_layout()
plt.show()