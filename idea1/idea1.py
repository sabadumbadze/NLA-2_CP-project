import numpy as np
import matplotlib.pyplot as plt


input_speed_rpm = [1500, 2200, 1800, 1200,
                   2800, 1600, 2100, 1300, 2500, 1900, 5000, 3000]
input_vibration_hz = [30,   55,   40,   25,
                      70,   35,   50,   28,   65,   42, 67, 80]
input_temperature = [50,   75,   60,   45,
                     85,   52,   70,   48,   80,   62, 100, 70]

target_wear_microns = [120, 210, 160, 100,
                       280, 135, 195, 110, 260, 175, 600, 400]


speed = np.array(input_speed_rpm)
vib = np.array(input_vibration_hz)
temp = np.array(input_temperature)
b = np.array(target_wear_microns)


num_samples = len(b)
bias = np.ones(num_samples)


A = np.column_stack((speed, vib, temp, bias))

print("--- Data Loaded ---")
print(f"Number of samples: {num_samples}")
print(f"Feature Matrix A shape: {A.shape}")
print("-" * 30)


def solve_ols(A, b):
    """
    Solves min ||Ax - b||^2 using NumPy's lstsq.
    This uses QR/SVD factorization under the hood for stability.
    """
    x_opt, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x_opt


x_est = solve_ols(A, b)

b_pred = A @ x_est

error = np.linalg.norm(b - b_pred)

print("--- Results ---")
print(f"Coefficients (Weights): {x_est}")
print(f"Residual Error (L2 Norm): {error:.4f}")


plt.figure(figsize=(10, 6))


plt.scatter(range(num_samples), b, color='blue',
            label='Actual Tool Wear', s=100)

plt.plot(range(num_samples), b_pred, color='red',
         linestyle='--', marker='x', label='OLS Predicted Wear')

plt.xlabel('Machine Sample ID')
plt.ylabel('Tool Wear value')
plt.title('OLS: Predictive Maintenance (Actual vs Predicted)')
plt.legend()
plt.grid(True)
plt.show()
