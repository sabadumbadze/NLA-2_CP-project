import numpy as np
import matplotlib.pyplot as plt


input_speed_rpm = [1500, 2200, 1800, 1200,
                   2800, 1600, 2100, 1300, 2500, 1900, 5000, 3000]

input_vibration_hz = [30,   55,   40,   25,
                      70,   35,   50,   28,   65,   42, 67, 80]

input_temperature = [50,   75,   60,   45,
                     85,   52,   70,   48,   80,   62, 100, 70]

target_wear_microns = [120, 210, 160, 100, 280, 135, 195, 110, 260, 175, 600, 400]


speed = np.array(input_speed_rpm)
vib = np.array(input_vibration_hz)
temp = np.array(input_temperature)
b = np.array(target_wear_microns)

num_samples = len(b)

bias = np.ones(num_samples)

# Create Design Matrix A
A = np.column_stack((speed, vib, temp, bias))

print("--- Data Loaded ---")
print(f"Samples: {num_samples}")
print("-" * 30)

# ==========================================
# 3. SOLVER (OLS ALGORITHM)
# ==========================================
def solve_ols(A, b):
    """
    Solves min ||Ax - b||^2
    Uses QR/SVD factorization internally (Standard for NLA)
    """
    x_opt, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x_opt


x_est = solve_ols(A, b)

b_pred = A @ x_est

error = np.linalg.norm(b - b_pred)

print(f"Calculated Weights: {x_est}")
print(f"Residual Error: {error:.4f}")


plt.figure(figsize=(10, 7))

plt.scatter(b, b_pred, color='blue', s=80, alpha=0.7, label='Data Points', edgecolors='k')


min_val = min(b.min(), b_pred.min())
max_val = max(b.max(), b_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=3, linestyle='--', label='Perfect Fit Line')


plt.vlines(b, b, b_pred, colors='gray', alpha=0.5, linestyle='dotted', label='Residuals (Errors)')

plt.title('Ordinary Least Squares: Actual vs Predicted Wear', fontsize=14)
plt.xlabel('Actual Tool Wear (Measured)', fontsize=12)
plt.ylabel('Predicted Tool Wear (Model Output)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig("ols_results.png")
plt.show()