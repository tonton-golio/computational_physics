# Eigenvalue Algorithms

---

## QR Algorithm

The **QR algorithm** is the standard method for computing **all eigenvalues** of a matrix.

### Basic QR Iteration

$$
\begin{aligned}
A_0 &= A \\
\text{For } k &= 0, 1, 2, \dots: \\
A_k &= Q_k R_k \quad \text{(QR factorization)} \\
A_{k+1} &= R_k Q_k
\end{aligned}
$$

The matrices $A_k$ converge to (quasi-)upper triangular form, with eigenvalues on the diagonal!

[[simulation qr-algorithm-animation]]

### Why It Works

Each iteration is a similarity transformation:

$$
A_{k+1} = R_k Q_k = Q_k^T Q_k R_k Q_k = Q_k^T A_k Q_k
$$

Similar matrices have the same eigenvalues, and upper triangular matrices have eigenvalues on the diagonal.

```python
def qr_algorithm(A, max_iter=1000, tol=1e-12):
    """
    Compute all eigenvalues using QR iteration.
    """
    n = len(A)
    A_k = A.copy()

    for i in range(max_iter):
        Q, R = np.linalg.qr(A_k)
        A_k = R @ Q

        # Check convergence (subdiagonal elements small)
        off_diag = np.sum(np.abs(np.tril(A_k, -1)))
        if off_diag < tol:
            break

    eigenvalues = np.diag(A_k)
    return eigenvalues, A_k, i + 1
```

### Practical Improvements

- **Hessenberg reduction**: Transform to upper Hessenberg form first (saves computation)
- **Shifts**: Use Wilkinson or Francis shifts for faster convergence
- **Implicit QR**: Avoid explicit QR factorization for efficiency

[[figure qr-convergence-plot]]

---

## Gershgorin Circle Theorem

A quick way to **bound eigenvalues** without computing them.

### Theorem

Every eigenvalue of $A$ lies within at least one **Gershgorin disc**:

$$
D_i = \left\{ z \in \mathbb{C} : |z - a_{ii}| \leq R_i \right\}
$$

where the radius is:

$$
R_i = \sum_{j \neq i} |a_{ij}|
$$

[[simulation gershgorin-circles]]

### Implementation

```python
def gershgorin(A):
    """
    Compute Gershgorin disc centers and radii.

    Every eigenvalue lies within at least one disc.
    """
    n = len(A)
    centers = np.array([A[i, i] for i in range(n)])
    radii = np.array([np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
                      for i in range(n)])
    return centers, radii
```

[[figure gershgorin-example]]

---

## Functions of Matrices

For non-defective $A = T\Lambda T^{-1}$, we can define functions of matrices:

$$
f(A) = T f(\Lambda) T^{-1} = T \begin{pmatrix} f(\lambda_1) & & \\ & \ddots & \\ & & f(\lambda_n) \end{pmatrix} T^{-1}
$$

### Applications

- **Matrix exponential**: $e^{At}$ solves $\frac{dx}{dt} = Ax$
- **Matrix logarithm**: Used in Lie groups
- **Matrix square root**: In covariance analysis

$$
P(A) = \sum_{k=0}^{d} c_k A^k = T \left(\sum_{k=0}^{d} c_k \Lambda^k\right) T^{-1} = T P(\Lambda) T^{-1}
$$

[[simulation matrix-exponential]]

---

## Algorithm Comparison

| Algorithm | Finds | Convergence | Complexity/iter | Best For |
|-----------|-------|-------------|-----------------|----------|
| Power Method | Dominant $\lambda$ | Linear | $O(n^2)$ | Single largest eigenvalue |
| Inverse Iteration | $\lambda$ near shift | Linear | $O(n^3)$ | Single eigenvalue with estimate |
| Rayleigh Quotient | Single $\lambda$ | Cubic | $O(n^3)$ | Fast convergence |
| QR Algorithm | All $\lambda$ | Quadratic | $O(n^3)$ | All eigenvalues |
| Lanczos | $k$ largest/smallest | Superlinear | $O(kn)$ | Large sparse matrices |

[[figure algorithm-comparison]]

---

## Summary

1. **Eigenvalues** characterize how linear transformations scale vectors
2. **Power method** finds the dominant eigenvalue with linear convergence
3. **Inverse iteration** finds eigenvalues near a given shift
4. **Rayleigh quotient iteration** achieves cubic convergence for Hermitian matrices
5. **QR algorithm** computes all eigenvalues and is the industry standard
6. **Gershgorin circles** provide quick eigenvalue bounds without computation

---

## References

1. Trefethen & Bau, *Numerical Linear Algebra*, Chapters 24-28
2. Golub & Van Loan, *Matrix Computations*, Chapter 7
3. Demmel, *Applied Numerical Linear Algebra*, Chapter 5
4. 3Blue1Brown: [Eigenvalues and Eigenvectors](https://www.youtube.com/watch?v=PFDu9oVAE-g)
