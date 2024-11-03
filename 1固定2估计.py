import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

file_path = "V_P_26-27.txt"
data = pd.read_csv(
    file_path,
    header=None,
    names=["Voltage", "Power"],
    skiprows=1,
    dtype={"Voltage": np.float64, "Power": np.float64},
)


# V1, P1 = data['Voltage'][34:66].values, data['Power'][34:66].values
V, P = data["Voltage"][1:].values, data["Power"][1:].values * 630


# 定义拟合函数
def model(v, a, b):
    return 0.5 * (1 - np.cos(a * v**2 + b))


# 使用 curve_fit 进行拟合
# 初始猜测 a 和 b 的值
initial_guess = [0.17, 3.7]
params, covariance = curve_fit(model, V, P, p0=initial_guess)
# params, covariance = curve_fit(model, V, P)
# 获取拟合参数
a_fitted, b_fitted = params
print(f"拟合得到的 a 值: {a_fitted}, b 值: {b_fitted}")

# 生成拟合结果
fitted_power = model(V, a_fitted, b_fitted)

# 创建更密集的电压点用于绘制平滑曲线
V_smooth = np.linspace(V.min(), V.max(), 1000)  # 增加到1000个点
fitted_power_smooth = model(V_smooth, a_fitted, b_fitted)

# 计算拟合优度 R²
residuals = P - fitted_power
ss_res = np.sum(residuals**2)
ss_tot = np.sum((P - np.mean(P)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R² = {r_squared:.6f}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(V, P, label="Original Data", color="blue", alpha=0.6, s=50)  # 增大散点大小
plt.plot(
    V_smooth, fitted_power_smooth, label="Fitted Curve", color="red", linewidth=2
)  # 增加线宽
plt.xlabel("Voltage (V)", fontsize=12)
plt.ylabel("Power (P)", fontsize=12)
plt.legend(fontsize=10)
plt.title(f"Curve Fitting (R² = {r_squared:.4f})", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tick_params(axis="both", which="major", labelsize=10)  # 调整刻度标签大小

# 调整图表边距
plt.tight_layout()

plt.show()
