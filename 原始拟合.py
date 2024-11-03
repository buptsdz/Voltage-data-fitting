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
V, P = data["Voltage"][1:33].values, data["Power"][1:33].values


# 定义拟合函数
def model(v, a, b):
    return 0.5 * (1 - np.cos(a * v**2 + b))


# 使用 curve_fit 进行拟合
# 初始猜测 a 和 b 的值
initial_guess = [0.1, 4]
params, covariance = curve_fit(model, V, P, p0=initial_guess)
# params, covariance = curve_fit(model, V, P)
# 获取拟合参数
a_fitted, b_fitted = params
print(f"拟合得到的 a 值: {a_fitted}, b 值: {b_fitted}")

# 生成拟合结果
fitted_power = model(V, a_fitted, b_fitted)

# 可视化结果
plt.scatter(V, P, label="DATA", color="blue")
plt.plot(V, fitted_power, label="FITTED", color="red")
plt.xlabel("V")
plt.ylabel("P")
plt.legend()
plt.title("FITTED")
plt.show()
