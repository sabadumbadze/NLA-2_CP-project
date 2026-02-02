
# 📊 Numerical Linear Algebra & Probabilistic Modeling

This project explores the intersection of **Numerical Linear Algebra (NLA)** and **Conditional Probability (CP)**. It focuses on implementing high-performance matrix algorithms to solve probabilistic models, specifically focusing on stochastic processes and network analysis.

## 🎯 Project Overview
The goal of this project is to apply advanced matrix factorization and iterative methods to analyze complex systems. By leveraging the power of linear algebra, we can efficiently compute stationary distributions in Markov Chains and determine the importance of nodes within a network.

## 🚀 Key Features
*   **Iterative Solvers:** Implementation of the **Power Iteration** method to find dominant eigenvalues and eigenvectors.
*   **Matrix Decomposition:** Exploration of matrix factorization techniques for dimensionality reduction and system stability analysis.
*   **Stochastic Modeling:** Computing transition probabilities using **Stochastic Matrices** and **Markov Chains**.
*   **Efficiency Optimization:** Utilizing **Sparse Matrix** computations to handle large-scale data with minimal memory overhead.
*   **Data Visualization:** Graphical representation of convergence rates and network rankings using Matplotlib.

## 🛠 Tech Stack
*   **Language:** Python 3.x
*   **Libraries:** 
    *   `NumPy`: High-performance multidimensional array operations.
    *   `SciPy`: Advanced linear algebra modules and sparse matrix handling.
    *   `Matplotlib`: Scientific data visualization.
*   **Environment:** Jupyter Notebook.

## 🧮 Mathematical Concepts Applied
The project implements several core NLA concepts, including:
*   **Eigenvalue Problems:** Solving $Ax = \lambda x$ for large-scale systems.
*   **Markov Chains:** Analyzing the behavior of random variables over time using probability transition matrices.
*   **Convergence Analysis:** Evaluating how damping factors ($\alpha$) affect the speed and stability of iterative algorithms.

## 📂 Project Structure
*   `*.ipynb`: Main Jupyter Notebooks containing the implementation and analysis.
*   `data/`: (Optional) Datasets used for testing the algorithms.
*   `.gitignore`: Configuration to keep the repository clean of temporary files.

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sabadumbadze/NLA-2_CP-project.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy scipy matplotlib
    ```
3.  **Run the analysis:**
    Launch Jupyter Notebook and open the main `.ipynb` file:
    ```bash
    jupyter notebook
    ```

