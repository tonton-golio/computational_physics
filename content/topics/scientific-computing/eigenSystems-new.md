# Eigenvalue Problems

Any stable physical/chemical system can be described with an eigensystem. Wavefunctions in quantum mechanics are a typical example.

$$
Ax = \lambda x
$$

Where $A$ is the operator (mapping $\mathbb{C}^n \rightarrow \mathbb{C}^n$), $x$ an eigenvector, and $\lambda$ an eigenvalue. The question is, **How do we obtain the eigen-vectors and -values?**

[[simulation eigen-basics]]

## The Characteristic Polynomial

For any fixed $\lambda$, we have a linear system:

$$
(A-\lambda I)x = 0
$$

This has non-trivial solutions ($x\neq0$) if and only if:

$$
\det(A-\lambda I) = 0
$$

[[figure characteristic-polynomial-visualization]]

## Eigenspaces and Multiplicity

Eigenvectors always come in subspaces. The **eigenspace** $E_\lambda$ is defined as:

$$
E_\lambda = \{x\in \mathbb{C}^n \mid Ax=\lambda x\}
$$

### Two Types of Multiplicity

1. **Algebraic multiplicity**: How many times an eigenvalue appears in the characteristic polynomial
2. **Geometric multiplicity**: The dimension of the eigenspace

[[simulation multiplicity-comparison]]

**Key insight**: Geometric multiplicity $\leq$ algebraic multiplicity

## The Power Method

Let $A:\mathbb{C}^n \rightarrow \mathbb{C}^n$ be a non-defective linear operator with $A=T\Lambda T^{-1}$.

Order eigenvalues: $|\lambda_1| \geq |\lambda_2| \geq \dots \geq |\lambda_n|$

[[simulation power-method-animation]]

### Power Method Algorithm

```python
def PowerIterate(A, x_0):
    x = x_0
    while (not_converged(A, x)):
        x = A @ x
        x /= np.linalg.norm(x)
    return x
```

### Convergence Analysis

After $k$ iterations:

$$
A^k x = \lambda_1^k \left(\tilde{x}_1 t_1 + O\left(\left|\frac{\lambda_2}{\lambda_1}\right|^k\right)\right)
$$

[[figure convergence-rate-plot]]

The ratio $|\lambda_2 / \lambda_1|$ determines convergence speed.

## Rayleigh Quotient Iteration

A more sophisticated approach using the **Rayleigh quotient**:

$$
\lambda_R(x) = \frac{x^T A x}{x^T x}
$$

[[simulation rayleigh-iteration]]

### Rayleigh Iteration Algorithm

```python
def rayleigh_iterate(A, x0, shift0, tol=0.001):
    norm = lambda v: np.max(abs(v))
    A = A.copy()
    A_less_eig = A - shift0 * np.eye(len(A))
    
    while not_converged:
        y = solve_linear_system(A_less_eig, x)
        c = (y @ x) / (x @ x)
        x = y / norm(y)
        lambda_new = 1/c + shift0
        
    return lambda_new, x
```

## Gershgorin Circle Theorem

A quick way to estimate eigenvalue locations without computing them.

[[simulation gershgorin-circles]]

For a matrix $A$, each eigenvalue lies within at least one Gershgorin disc:

$$
D_i = \left\{ z \in \mathbb{C} : |z - a_{ii}| \leq \sum_{j \neq i} |a_{ij}| \right\}
$$

[[figure gershgorin-discs-example]]

## Applications

### Quantum Mechanics
Stationary states are eigenvectors of the Hamiltonian operator.

[[simulation quantum-eigenstates]]

### Vibrating Systems
Natural frequencies and mode shapes of structures.

[[simulation vibrating-modes]]

### Principal Component Analysis
Covariance matrix eigenvectors give principal directions.

[[simulation pca-demo]]

---

## Summary

| Method | Finds | Convergence | Complexity |
|--------|-------|-------------|------------|
| Power Method | Dominant eigenpair | $O(|\lambda_2/\lambda_1|^k)$ | $O(n^2)$ per iteration |
| Rayleigh Iteration | Any eigenpair | Cubic (nearby) | $O(n^3)$ per iteration |
| QR Algorithm | All eigenpairs | Cubic | $O(n^3)$ |
