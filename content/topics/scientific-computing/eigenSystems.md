# Eigenvalue Problems

Any stable physical/chemical system can be described with an eigensystem. Wavefunctions in quantum mechanics are a typical example.

$$
Ax = \\lambda x
$$

Where $A$ is the operator (mapping $\\mathbb{C}^n \\rightarrow \\mathbb{C}^n$), $x$ an eigenvector, and $\\lambda$ an eigenvalue. The question is: **How do we obtain the eigenvalues and eigenvectors?**

## Why Eigenvalues Matter

Eigenvalues appear throughout physics and engineering:

- **Quantum mechanics**: Energy levels are eigenvalues of the Hamiltonian
- **Vibrations**: Natural frequencies of structures
- **Stability analysis**: System behavior near equilibrium points
- **Principal Component Analysis**: Dimensionality reduction in data science
- **Google PageRank**: Largest eigenvector of the web graph

---

## Applications

### Quantum Mechanics
Stationary states are eigenvectors of the Hamiltonian operator.

### Vibrating Systems
Natural frequencies and mode shapes of structures.

### Principal Component Analysis
Covariance matrix eigenvectors give principal directions.

```python
import numpy as np

# Simple PCA example
np.random.seed(42)
data = np.random.randn(100, 2) @ np.array([[2, 1], [1, 3]])  # correlated data
cov = np.cov(data.T)
evals, evecs = np.linalg.eig(cov)
print('Variance explained:', evals / np.sum(evals))
print('Principal direction:', evecs[:, np.argmax(evals)])
```

---

## Mathematical Foundations

### The Characteristic Polynomial

For any fixed $\\lambda$, we have a linear system:

$$
(A-\\lambda I)x = 0
$$

This has **non-trivial** solutions ($x\\neq0$) if and only if:

$$
\\det(A-\\lambda I) = 0
$$

### Eigenspaces and Multiplicity

Eigenvectors form subspaces called **eigenspaces**:

$$
E_\\lambda = \\{x\\in \\mathbb{C}^n \\mid Ax=\\lambda x\\}
$$

The characteristic polynomial $P(\\lambda) = \\det(A-\\lambda I)$ is of degree $n$, and by the **fundamental theorem of algebra**, has exactly $n$ complex roots (counting multiplicity).

**Two types of multiplicity:**

1. **Algebraic multiplicity**: How many times $\\lambda_i$ appears in the characteristic polynomial
2. **Geometric multiplicity**: Dimension of the eigenspace $E_\\lambda$

**Key inequality**: Geometric multiplicity $\\leq$ algebraic multiplicity

### Defective vs Non-Defective Matrices

- **Non-defective**: $\\sum_{\\lambda\\in Sp(A)} \\dim(E_\\lambda) = n$ — we have a complete eigenbasis
- **Defective**: $\\sum_{\\lambda\\in Sp(A)} \\dim(E_\\lambda) < n$ — not enough eigenvectors

**Non-defectiveness is guaranteed if:**
- $A$ is **normal**: $A^H A = AA^H$
- All eigenvalues are distinct
- $A$ is **Hermitian**: $A=A^H$ (best case — guarantees orthonormal eigenbasis)

For Hermitian matrices: $A = U \\Lambda U^H$

---

## The Power Method

Let $A$ be non-defective with $A=T\\Lambda T^{-1}$. Order the eigenvalues by magnitude:

$$
|\\lambda_1| \\geq |\\lambda_2| \\geq \\dots \\geq |\\lambda_n|
$$

### Derivation

For any $x \\in \\mathbb{C}^n$, expressed in the eigenbasis:

$$
x = \\tilde{x}_1 t_1 + \\tilde{x}_2 t_2 + \\dots + \\tilde{x}_n t_n
$$

Applying $A$ repeatedly:

$$
A^k x = \\tilde{x}_1 \\lambda_1^k t_1 + \\tilde{x}_2 \\lambda_2^k t_2 + \\dots + \\tilde{x}_n \\lambda_n^k t_n
$$

$$
= \\lambda_1^k \\left(\\tilde{x}_1 t_1 + \\tilde{x}_2 \\left(\\frac{\\lambda_2}{\\lambda_1}\\right)^k t_2 + \\dots + \\tilde{x}_n \\left(\\frac{\\lambda_n}{\\lambda_1}\\right)^k t_n\\right)
$$

Since $|\\lambda_1|$ is largest, the ratios approach zero:

$$
\\lim_{k\\to\\infty}\\frac{A^k x}{\\|A^k x\\|} = t_1, \\quad \\text{if } \\tilde{x}_1 \\neq 0 \\text{ and } |\\lambda_2| < |\\lambda_1|
$$

### Power Method Algorithm

```python
def power_iterate(A, x0, tol=1e-10, max_iter=1000):
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

### Example with Convergence Visualization

```python
import matplotlib.pyplot as plt

def power_demo(A, max_iter=30):
    np.random.seed(42)
    x0 = np.random.rand(*A.shape)
    residuals = []
    x = x0.flatten() / np.linalg.norm(x0)
    true_dom = np.max(np.linalg.eigvals(A))
    for i in range(max_iter):
        y = A @ x
        x_new = y / np.linalg.norm(y)
        eigenvalue = x_new @ A @ x_new
        res = np.abs(eigenvalue - true_dom)
        residuals.append(res)
        if res < 1e-8:
            break
        x = x_new
    plt.figure()
    plt.semilogy(residuals)
    plt.xlabel('# Iterations')
    plt.ylabel('Error in λ')
    plt.title('Power Method Convergence')
    plt.grid(True)
    plt.show()

# Test matrix
A = np.array([[4, 1], [1, 2]])
power_demo(A)
```

### Limitations

- Only finds the **dominant eigenvalue** (largest magnitude)
- Fails if the dominant eigenvalue is not unique
- Slow convergence when $|\\lambda_1| \\approx |\\lambda_2|$ (small spectral gap)

---

## Inverse Iteration

To find an eigenvalue **closest to a shift** $\\sigma$, apply the power method to $(A - \\sigma I)^{-1}$:

The eigenvalues of $(A - \\sigma I)^{-1}$ are $(\\lambda_i - \\sigma)^{-1}$, so the one closest to $\\sigma$ becomes dominant.

```python
def inverse_iterate(A, sigma, x0, tol=1e-10, max_iter=1000):
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

### Example Visualization

```python
def inverse_demo(A, sigma=None, max_iter=20):
    if sigma is None:
        sigma = np.trace(A)/len(A)  # rough guess
    np.random.seed(42)
    x0 = np.random.rand(*A.shape)
    residuals = []
    x = x0.flatten() / np.linalg.norm(x0)
    evals = np.linalg.eigvals(A)
    target_lambda = evals[np.argmin(np.abs(evals - sigma))]
    for i in range(max_iter):
        n = len(A)
        A_shift = A - sigma * np.eye(n)
        y = np.linalg.solve(A_shift, x)
        x_new = y / np.linalg.norm(y)
        eigenvalue = x_new @ A @ x_new
        res = np.abs(eigenvalue - target_lambda)
        residuals.append(res)
        x = x_new
        sigma = eigenvalue  # update shift?
    plt.semilogy(residuals)
    plt.xlabel('# Iterations')
    plt.ylabel('Error in λ')
    plt.title('Inverse Iteration Convergence')
    plt.grid(True)
    plt.show()

A = np.diag([1,4,10])
inverse_demo(A, sigma=4)
```

---

## Rayleigh Quotient Iteration

... keep as is, add similar demo if space.

## QR Algorithm

Keep, enhance.

### Basic QR Iteration

... code

Add example:

```python
def qr_demo(A, max_iter=50):
    Ak = A.copy().astype(float)
    off_diags = []
    for i in range(max_iter):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
        off = np.linalg.norm(np.tril(Ak, -1))
        off_diags.append(off)
        if off < 1e-12:
            break
    evals_true = np.linalg.eigvals(A)
    evals_comp = np.diag(Ak)
    print('True:', evals_true)
    print('Computed:', evals_comp)
    plt.figure()
    plt.semilogy(off_diags)
    plt.xlabel('Iterations')
    plt.ylabel('Off-diagonal norm')
    plt.title('QR Algorithm Convergence')
    plt.grid(True)
    plt.show()
    return evals_comp

# Symmetric test
A = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
qr_demo(A)
```

Yes.

Then keep rest.

For Gershgorin, add plot demo.

```python
def plot_gershgorin_demo(A):
    centers, radii = gershgorin(A)
    fig, ax = plt.subplots(figsize=(6,6))
    for c, r in zip(centers, radii):
        circle = plt.Circle((c.real, c.imag), r, color='blue', fill=False, alpha=0.7)
        ax.add_patch(circle)
        ax.plot(c.real, c.imag, 'bo')
    evals = np.linalg.eigvals(A)
    ax.plot(evals.real, evals.imag, 'rx', markersize=10)
    ax.set_aspect('equal')
    ax.set_title('Gershgorin Circles (X: eigenvalues)')
    ax.grid(True)
    plt.show()

A_complex = np.array([[1+1j, 2], [0.5, 3-1j]])
plot_gershgorin_demo(A_complex)
```

Perfect.

Add this after gersh code.

For other sections, similar if needed.

Remove the summary if duplicate, keep main one.

Now, since content long, but write allows.

Also, for matrix functions, add brief.

Now, write the file with these enhancements.

To fit, I have abbreviated above.

For the response, since subagent final is report.

But first write.

The content parameter is long, but ok.

Let me complete the content string.

Also, add to end the references.

For cal log, after commit.

First write the md.<|control12|><xai:function_call name="write">
<parameter name="path">/Users/koalaclaw/Documents/claw/koala-brain/content/topics/scientific-computing/eigenSystems.md