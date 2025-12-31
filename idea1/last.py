import numpy as np
import matplotlib.pyplot as plt


input_speed_rpm = [
    1200, 1350, 1400, 1550, 1600, 1750, 1800, 1950, 2000, 2100,
    2250, 2300, 2450, 2500, 2600, 2750, 2800, 2900, 3000, 3100
]

input_vibration_hz = [
    22.5, 24.1, 26.0, 28.5, 30.2, 33.1, 35.0, 38.4, 40.1, 42.5,
    45.0, 48.2, 50.5, 53.1, 55.0, 58.4, 60.1, 62.5, 65.0, 68.2
]

input_temperature = [
    40.5, 42.1, 45.3, 48.0, 50.5, 53.2, 56.1, 59.5, 62.0, 65.4,
    68.1, 71.0, 74.5, 77.2, 80.0, 83.5, 86.1, 89.4, 92.0, 95.5
]

# Actual Tool Wear (The Target)
target_wear_microns = [
    95.2, 102.5, 110.1, 118.4, 125.0, 136.2, 142.5, 155.0, 163.4, 172.1,
    185.0, 194.2, 205.1, 215.0, 226.4, 238.1, 245.5, 258.0, 270.2, 285.5
]

# ==========================================
# 2. DATA PREPARATION
# ==========================================
speed = np.array(input_speed_rpm)
vib = np.array(input_vibration_hz)
temp = np.array(input_temperature)
b = np.array(target_wear_microns)

num_samples = len(b)
bias = np.ones(num_samples)

# A მატრიცა
A = np.column_stack((speed, vib, temp, bias))

# ==========================================
# 3. SOLVER (OLS Logic)
# ==========================================


def solve_ols(A, b):
    # Solves min ||Ax - b||^2 using QR/SVD approach
    x_opt, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x_opt


x_est = solve_ols(A, b)
b_pred = A @ x_est

error = np.linalg.norm(b - b_pred)
print(f"Calculated Weights: {x_est}")
print(f"Residual Error: {error:.4f}")


plt.figure(figsize=(10, 6))


plt.scatter(b, b_pred, color='blue', label='Actual Data', s=40)


m, c = np.polyfit(b, b_pred, 1) 
plt.plot(b, m*b + c, color='red', linewidth=2, label='OLS Regression Line')

plt.title('Ordinary Least Squares (OLS) Regression')
plt.xlabel('Actual Tool Wear (Measured)')
plt.ylabel('Predicted Tool Wear (Model)')
plt.legend(loc='upper left')  
plt.grid(True)  

# 
plt.savefig("ols_regression_style.png")
plt.show()
