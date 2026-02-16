# Eigenvalue Problems

Any stable physical/chemical system can be described with an eigensystem. Wavefunctions in quantum mechanics are a typical example.

$$
Ax = \lambda x
$$

Where $A$ is the operator (mapping $\mathbb{C}^n \rightarrow \mathbb{C}^n$), $x$ an eigenvector, and $\lambda$ an eigenvalue. The question is: **How do we obtain the eigenvalues and eigenvectors?**

[[simulation eigen-transformation]]

## Why Eigenvalues Matter

Eigenvalues appear throughout physics and engineering:

- **Quantum mechanics**: Energy levels are eigenvalues of the Hamiltonian
- **Vibrations**: Natural frequencies of structures
- **Stability analysis**: System behavior near equilibrium points
- **Principal Component Analysis**: Dimensionality reduction in data science
- **Google PageRank**: Largest eigenvector of the web graph

[[figure eigen-applications]]

---

## Mathematical Foundations

### The Characteristic Polynomial

For any fixed $\lambda$, we have a linear system:

$$
(A-\lambda I)x = 0
$$

This has **non-trivial** solutions ($x\neq0$) if and only if:

$$
\det(A-\lambda I) = 0
$$

[[simulation characteristic-polynomial]]

### Eigenspaces and Multiplicity

Eigenvectors form subspaces called **eigenspaces**:

$$
E_\lambda = \{x\in \mathbb{C}^n \mid Ax=\lambda x\}
$$

The characteristic polynomial $P(\lambda) = \det(A-\lambda I)$ is of degree $n$, and by the **fundamental theorem of algebra**, has exactly $n$ complex roots (counting multiplicity).

**Two types of multiplicity:**

1. **Algebraic multiplicity**: How many times $\lambda_i$ appears in the characteristic polynomial
2. **Geometric multiplicity**: Dimension of the eigenspace $E_\lambda$

[[figure multiplicity-diagram]]

**Key inequality**: Geometric multiplicity $\leq$ algebraic multiplicity

### Defective vs Non-Defective Matrices

- **Non-defective**: $\sum_{\lambda\in Sp(A)} \dim(E_\lambda) = n$ — we have a complete eigenbasis
- **Defective**: $\sum_{\lambda\in Sp(A)} \dim(E_\lambda) < n$ — not enough eigenvectors

**Non-defectiveness is guaranteed if:**
- $A$ is **normal**: $A^H A = AA^H$
- All eigenvalues are distinct
- $A$ is **Hermitian**: $A=A^H$ (best case — guarantees orthonormal eigenbasis)

For Hermitian matrices: $A = U \Lambda U^H$

[[simulation hermitian-demo]]

---

## The Power Method

Let $A$ be non-defective with $A=T\Lambda T^{-1}$. Order the eigenvalues by magnitude:

$$
|\lambda_1| \geq |\lambda_2| \geq \dots \geq |\lambda_n|
$$

### Derivation

For any $x \in \mathbb{C}^n$, expressed in the eigenbasis:

$$
x = \tilde{x}_1 t_1 + \tilde{x}_2 t_2 + \dots + \tilde{x}_n t_n
$$

Applying $A$ repeatedly:

$$
A^k x = \tilde{x}_1 \lambda_1^k t_1 + \tilde{x}_2 \lambda_2^k t_2 + \dots + \tilde{x}_n \lambda_n^k t_n
$$

$$
= \lambda_1^k \left(\tilde{x}_1 t_1 + \tilde{x}_2 \left(\frac{\lambda_2}{\lambda_1}\right)^k t_2 + \dots + \tilde{x}_n \left(\frac{\lambda_n}{\lambda_1}\right)^k t_n\right)
$$

Since $|\lambda_1|$ is largest, the ratios approach zero:

$$
\lim_{k\to\infty}\frac{A^k x}{\|A^k x\|} = t_1, \quad \text{if } \tilde{x}_1 \neq 0 \text{ and } |\lambda_2| < |\lambda_1|
$$

[[simulation power-method-animation]]

### Power Method Algorithm

```python
def power_iterate(A, x0, tol=1e-10, max_iter=1000):
    """
    Find the dominant eigenvalue and eigenvector.
    
    Parameters:
        A: Square matrix
        x0: Initial guess vector
        tol: Convergence tolerance
        max_iter: Maximum iterations
    
    Returns:
        eigenvalue, eigenvector, iterations
    """
    x = x0 / np.linalg.norm(x0)
    
    for i in range(max_iter):
        y = A @ x
        x_new = y / np.linalg.norm(y)
        
        # Rayleigh quotient for eigenvalue estimate
        eigenvalue = x_new @ A @ x_new
        
        if np.linalg.norm(x_new - x) < tol or np.linalg.norm(x_new + x) < tol:
            return eigenvalue, x_new, i + 1
        
        x = x_new
    
    return eigenvalue, x, max_iter
```

### Limitations

- Only finds the **dominant eigenvalue** (largest magnitude)
- Fails if the dominant eigenvalue is not unique
- Slow convergence when $|\lambda_1| \approx |\lambda_2|$ (small spectral gap)

[[figure convergence-comparison]]

---

## Inverse Iteration

To find an eigenvalue **closest to a shift** $\sigma$, apply the power method to $(A - \sigma I)^{-1}$:

The eigenvalues of $(A - \sigma I)^{-1}$ are $(\lambda_i - \sigma)^{-1}$, so the one closest to $\sigma$ becomes dominant.

```python
def inverse_iterate(A, sigma, x0, tol=1e-10, max_iter=1000):
    """
    Find eigenvalue closest to sigma.
    """
    n = len(A)
    x = x0 / np.linalg.norm(x0)
    A_shifted = A - sigma * np.eye(n)
    
    for i in range(max_iter):
        # Solve (A - sigma*I) y = x
        y = np.linalg.solve(A_shifted, x)
        x_new = y / np.linalg.norm(y)
        
        eigenvalue = x_new @ A @ x_new
        
        if np.linalg.norm(x_new - x) < tol or np.linalg.norm(x_new + x) < tol:
            return eigenvalue, x_new, i + 1
        
        x = x_new
    
    return eigenvalue, x, max_iter
```

[[simulation inverse-iteration]]

---

## Rayleigh Quotient Iteration

The **Rayleigh quotient** provides an eigenvalue estimate:

$$
\lambda_R(x) = \frac{x^T A x}{x^T x}
$$

Rayleigh quotient iteration updates the shift at each step, achieving **cubic convergence** for Hermitian matrices!

```python
def rayleigh_quotient_iteration(A, x0, tol=1e-12, max_iter=100):
    """
    Fast cubic convergence for Hermitian matrices.
    """
    n = len(A)
    x = x0 / np.linalg.norm(x0)
    eigenvalue = x @ A @ x
    
    for i in range(max_iter):
        A_shifted = A - eigenvalue * np.eye(n)
        
        try:
            y = np.linalg.solve(A_shifted, x)
        except np.linalg.LinAlgError:
            # Singular matrix means we hit an eigenvalue
            return eigenvalue, x, i
        
        x_new = y / np.linalg.norm(y)
        eigenvalue_new = x_new @ A @ x_new
        
        if abs(eigenvalue_new - eigenvalue) < tol:
            return eigenvalue_new, x_new, i + 1
        
        x = x_new
        eigenvalue = eigenvalue_new
    
    return eigenvalue, x, max_iter
```

[[simulation rayleigh-convergence]]

### Convergence Comparison

| Method | Convergence Rate | Finds |
|--------|------------------|-------|
| Power Method | Linear | Dominant $\lambda$ |
| Inverse Iteration | Linear | $\lambda$ near shift |
| Rayleigh Quotient Iteration | **Cubic** (Hermitian) | Single $\lambda$ |

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
