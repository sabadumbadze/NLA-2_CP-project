import numpy as np
import matplotlib.pyplot as plt


A_data = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.1, 0.0, 0.0],
    [0.0, 0.0, 2.5, 0.0],
    [0.0, 0.0, 0.0, 0.8]
]

b_data = [180, 150, 60, 200]

C_data = [[1, 1, 1, 1]]

d_data = [1000]

A = np.array(A_data)
b = np.array(b_data)
C = np.array(C_data)
d = np.array(d_data)

print("--- Data Loaded ---")
print(f"Ideal Targets (Sum): {np.sum(b)}")
print(f"Required Demand (Constraint): {d[0]}")
print("-" * 30)


def solve_constrained_ls(A, b, C, d):
    """
    Constructs and solves the KKT Block Matrix system.
    """
    ATA = A.T @ A
    ATb = A.T @ b

    row1 = np.hstack((2 * ATA, C.T))

    p = C.shape[0]
    row2 = np.hstack((C, np.zeros((p, p))))

    KKT_Matrix = np.vstack((row1, row2))

    rhs = np.concatenate((2 * ATb, d))

    result = np.linalg.solve(KKT_Matrix, rhs)

    return result[:A.shape[1]]


x_alloc = solve_constrained_ls(A, b, C, d)

print("--- Results ---")
print(f"Calculated Allocation (x): {x_alloc}")
print(f"Sum of Allocation: {np.sum(x_alloc):.4f}")
print(f"Constraint Satisfied? {np.isclose(np.sum(x_alloc), d[0])}")


labels = ['Solar', 'Wind', 'Gas', 'Hydro']
x_pos = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x_pos - width/2, b, width, label='Ideal Target (b)', color='skyblue')

plt.bar(x_pos + width/2, x_alloc, width,
        label='Constrained Output (x)', color='orange')

plt.xlabel('Power Plants')
plt.ylabel('Generation (MW)')
plt.title(f'Energy Dispatch: Meeting {d[0]}MW Demand')
plt.xticks(x_pos, labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

for i, v in enumerate(x_alloc):
    plt.text(i + width/2, v + 2, f"{v:.1f}", ha='center', fontweight='bold')

plt.savefig("cls_results.png")
plt.show()
