# Eigenvalue Problems
*Discovering the secret personalities hiding inside matrices*

> *Every matrix has a set of special directions that it only stretches — never rotates. These are the eigenvectors, and the stretch factors are the eigenvalues. Together, they reveal the DNA of the matrix.*

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

*Every time you ask "what are the natural modes of this system?" you're asking an eigenvalue question.*

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

*This says: eigenvalues are the values of $\lambda$ that make the matrix $(A - \lambda I)$ singular. Setting the determinant to zero gives us a polynomial to solve.*

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

> **You might be wondering...** "Why do we care if a matrix is defective?" Because if it's non-defective, we can diagonalize it — write $A = T\Lambda T^{-1}$ — which makes everything easy. Computing $A^{100}$? Just compute $\lambda_i^{100}$. Solving $e^{At}$? Just compute $e^{\lambda_i t}$. Defective matrices are the troublemakers that make life harder.

[[simulation hermitian-demo]]

---

## Part I: Simple Methods (finding one eigenvalue at a time)

### The Power Method — the world's laziest way to find the boss eigenvalue

Here's the idea: take any vector and multiply it by $A$ over and over. The dominant eigenvalue's eigenvector will gradually take over, like the loudest voice in a room eventually drowning out everyone else.

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

Since $|\lambda_1|$ is strictly the largest, the ratios approach zero:

$$
\lim_{k\to\infty}\frac{A^k x}{\|A^k x\|} = t_1, \quad \text{if } \tilde{x}_1 \neq 0 \text{ and } |\lambda_2| < |\lambda_1|
$$

*This says: keep multiplying and normalizing. The biggest eigenvalue wins because its ratio keeps growing while all others shrink to zero. Convergence speed depends on how much bigger $\lambda_1$ is than $\lambda_2$.*

**Important**: This requires the dominant eigenvalue to be strictly larger in magnitude than the second largest. If $|\lambda_1| = |\lambda_2|$, the method fails to converge. Convergence rate is $O(|\lambda_2/\lambda_1|^k)$, so a small spectral gap means slow convergence.

[[simulation power-method-animation]]

### Power Method Algorithm

```python
def power_iterate(A, x0, tol=1e-10, max_iter=1000):
    """
    Find the dominant eigenvalue and eigenvector.
    The laziest eigenvalue finder: just keep multiplying and normalizing.
    """
    x = x0 / np.linalg.norm(x0)       # start with any unit vector

    for i in range(max_iter):
        y = A @ x                       # multiply by A
        x_new = y / np.linalg.norm(y)   # normalize so it doesn't blow up

        # Rayleigh quotient for eigenvalue estimate
        eigenvalue = x_new.conj() @ A @ x_new

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

### Inverse Iteration — finding eigenvalues near a target

To find an eigenvalue **closest to a shift** $\sigma$, apply the power method to $(A - \sigma I)^{-1}$:

The eigenvalues of $(A - \sigma I)^{-1}$ are $(\lambda_i - \sigma)^{-1}$, so the one closest to $\sigma$ becomes dominant.

*This is the beautiful trick: flip the problem inside-out. The eigenvalue you care about (near $\sigma$) becomes the dominant one of the inverse.*

```python
def inverse_iterate(A, sigma, x0, tol=1e-10, max_iter=1000):
    """
    Find eigenvalue closest to sigma.
    Flip the spectrum so the one you want becomes dominant.
    """
    n = len(A)
    x = x0 / np.linalg.norm(x0)
    A_shifted = A - sigma * np.eye(n)  # shift to make target eigenvalue smallest

    for i in range(max_iter):
        y = np.linalg.solve(A_shifted, x)  # solve instead of invert
        x_new = y / np.linalg.norm(y)

        eigenvalue = x_new @ A @ x_new

        if np.linalg.norm(x_new - x) < tol or np.linalg.norm(x_new + x) < tol:
            return eigenvalue, x_new, i + 1

        x = x_new

    return eigenvalue, x, max_iter
```

[[simulation inverse-iteration]]

---

### Rayleigh Quotient Iteration — cubic convergence

The **Rayleigh quotient** provides an eigenvalue estimate:

$$
\lambda_R(x) = \frac{x^T A x}{x^T x}
$$

*This says: project $A$ onto the direction of $x$ to get your best guess for the eigenvalue. It's the average "stretch factor" along that direction.*

Rayleigh quotient iteration updates the shift at each step, achieving **cubic convergence** for Hermitian matrices!

```python
def rayleigh_quotient_iteration(A, x0, tol=1e-12, max_iter=100):
    """
    Fast cubic convergence for Hermitian matrices.
    The shift tracks the eigenvalue, so convergence accelerates dramatically.
    """
    n = len(A)
    x = x0 / np.linalg.norm(x0)
    eigenvalue = x @ A @ x            # initial Rayleigh quotient

    for i in range(max_iter):
        A_shifted = A - eigenvalue * np.eye(n)

        try:
            y = np.linalg.solve(A_shifted, x)
        except np.linalg.LinAlgError:
            return eigenvalue, x, i    # singular means we nailed the eigenvalue

        x_new = y / np.linalg.norm(y)
        eigenvalue_new = x_new @ A @ x_new

        if abs(eigenvalue_new - eigenvalue) < tol:
            return eigenvalue_new, x_new, i + 1

        x = x_new
        eigenvalue = eigenvalue_new

    return eigenvalue, x, max_iter
```

> **You might be wondering...** "How can convergence be *cubic*? That's insane." It is insane — and beautiful. The Rayleigh quotient is an optimally accurate eigenvalue estimate for a given eigenvector approximation. As the eigenvector improves, the shift gets better, which makes the inverse iteration converge faster, which improves the eigenvector faster... It's a virtuous cycle that accelerates explosively.

[[simulation rayleigh-convergence]]

### Convergence Comparison

| Method | Convergence Rate | Finds | Why it feels like magic |
|--------|------------------|-------|------------------------|
| Power Method | Linear | Dominant $\lambda$ | Just multiply and wait — the biggest voice wins |
| Inverse Iteration | Linear | $\lambda$ near shift | Flip the spectrum so your target becomes dominant |
| Rayleigh Quotient Iteration | **Cubic** (Hermitian) | Single $\lambda$ | The shift chases the eigenvalue in a virtuous spiral |

---

## Part II: Fancy Methods (finding all eigenvalues at once)

### QR Algorithm

The **QR algorithm** is the standard method for computing **all eigenvalues** of a matrix. It's the industry workhorse — the method behind `numpy.linalg.eig`.

#### Basic QR Iteration

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

#### Why It Works

Each iteration is a similarity transformation:

$$
A_{k+1} = R_k Q_k = Q_k^T Q_k R_k Q_k = Q_k^T A_k Q_k
$$

*This says: we're rotating the matrix without changing its eigenvalues (similarity transformation), and the rotation is chosen to gradually push the off-diagonal entries toward zero. When they vanish, the eigenvalues appear on the diagonal.*

Similar matrices have the same eigenvalues, and upper triangular matrices have eigenvalues on the diagonal.

```python
def qr_algorithm(A, max_iter=1000, tol=1e-12):
    """
    Compute all eigenvalues using QR iteration.
    Factorize, reverse-multiply, repeat until eigenvalues appear on the diagonal.
    """
    n = len(A)
    A_k = A.copy()

    for i in range(max_iter):
        Q, R = np.linalg.qr(A_k)   # factorize into orthogonal * upper triangular
        A_k = R @ Q                 # reverse-multiply: this is the magic step

        # Check convergence: are the subdiagonal elements small?
        off_diag = np.sum(np.abs(np.tril(A_k, -1)))
        if off_diag < tol:
            break

    eigenvalues = np.diag(A_k)
    return eigenvalues, A_k, i + 1
```

#### Practical Improvements

- **Hessenberg reduction**: Transform to upper Hessenberg form first (saves computation)
- **Shifts**: Use Wilkinson or Francis shifts for faster convergence
- **Implicit QR**: Avoid explicit QR factorization for efficiency

[[figure qr-convergence-plot]]

---

### Gershgorin Circle Theorem

A quick way to **bound eigenvalues** without computing them — like getting a rough map before starting the hike.

#### Theorem

Every eigenvalue of $A$ lies within at least one **Gershgorin disc**:

$$
D_i = \left\{ z \in \mathbb{C} : |z - a_{ii}| \leq R_i \right\}
$$

where the radius is:

$$
R_i = \sum_{j \neq i} |a_{ij}|
$$

*This says: draw a circle around each diagonal entry, with radius equal to the sum of the absolute values of the other entries in that row. Every eigenvalue lives inside at least one of these circles.*

[[simulation gershgorin-circles]]

#### Implementation

```python
def gershgorin(A):
    """
    Compute Gershgorin disc centers and radii.
    Quick and dirty eigenvalue bounds — no iteration needed!
    """
    n = len(A)
    centers = np.array([A[i, i] for i in range(n)])
    radii = np.array([np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
                      for i in range(n)])
    return centers, radii
```

[[figure gershgorin-example]]

> **Challenge:** Make a random 4x4 matrix, compute its Gershgorin circles, and plot them as circles in the complex plane. Then compute the actual eigenvalues with `np.linalg.eig` and plot them as dots. Are they all inside the circles? (They must be!)

---

### Functions of Matrices

For **non-defective** $A = T\Lambda T^{-1}$ (i.e., $A$ has a complete eigenbasis), we can define functions of matrices:

$$
f(A) = T f(\Lambda) T^{-1} = T \begin{pmatrix} f(\lambda_1) & & \\ & \ddots & \\ & & f(\lambda_n) \end{pmatrix} T^{-1}
$$

*This says: to apply a function to a matrix, diagonalize it, apply the function to each eigenvalue separately, and transform back. Want $e^A$? Compute $e^{\lambda_i}$ for each eigenvalue.*

#### Applications

- **Matrix exponential**: $e^{At}$ solves $\frac{dx}{dt} = Ax$
- **Matrix logarithm**: Used in Lie groups
- **Matrix square root**: In covariance analysis

$$
P(A) = \sum_{k=0}^{d} c_k A^k = T \left(\sum_{k=0}^{d} c_k \Lambda^k\right) T^{-1} = T P(\Lambda) T^{-1}
$$

For defective matrices, the Jordan normal form extends this framework, but the eigendecomposition $T\Lambda T^{-1}$ no longer exists. In practice, most matrices encountered in physical applications (Hermitian operators, normal matrices) are non-defective.

Matrix functions connect eigenvalue theory to dynamical systems (via $e^{At}$), quantum mechanics (operator functions of the Hamiltonian), and numerical methods for ODEs (matrix exponential integrators).

> *Remember stability from Lesson 10 (ODEs)? The eigenvalues of the Jacobian determine whether a system is stiff. The matrix exponential $e^{At}$ is exactly how ODE solutions evolve.*

[[simulation matrix-exponential]]

---

## Full Algorithm Comparison

| Algorithm | Finds | Convergence | Complexity/iter | Best For | Why it feels like magic |
|-----------|-------|-------------|-----------------|----------|------------------------|
| Power Method | Dominant $\lambda$ | Linear | $O(n^2)$ | Single largest eigenvalue | The loudest voice wins |
| Inverse Iteration | $\lambda$ near shift | Linear | $O(n^3)$ | Single eigenvalue with estimate | Flip the spectrum inside-out |
| Rayleigh Quotient | Single $\lambda$ | Cubic | $O(n^3)$ | Fast convergence | The shift chases the eigenvalue |
| QR Algorithm | All $\lambda$ | Quadratic | $O(n^3)$ | All eigenvalues | Factorize and reverse-multiply |
| Lanczos | $k$ largest/smallest | Superlinear | $O(kn)$ | Large sparse matrices | Only touch what matters |

[[figure algorithm-comparison]]

---

## Summary

1. **Eigenvalues** characterize how linear transformations scale vectors — they're the personality of the matrix
2. **Power method** finds the dominant eigenvalue with linear convergence — the laziest approach that works
3. **Inverse iteration** finds eigenvalues near a given shift by flipping the spectrum
4. **Rayleigh quotient iteration** achieves cubic convergence for Hermitian matrices through a brilliant feedback loop
5. **QR algorithm** computes all eigenvalues and is the industry standard behind every linear algebra library
6. **Gershgorin circles** provide quick eigenvalue bounds without computation — your rough map before the detailed survey

---

## References

1. Trefethen & Bau, *Numerical Linear Algebra*, Chapters 24-28
2. Golub & Van Loan, *Matrix Computations*, Chapter 7
3. Demmel, *Applied Numerical Linear Algebra*, Chapter 5
4. 3Blue1Brown: [Eigenvalues and Eigenvectors](https://www.youtube.com/watch?v=PFDu9oVAE-g) — watch this first for visual intuition

---

**What we just learned in one sentence:** Eigenvalues reveal how matrices stretch space along special directions, and we can find them lazily (power method), precisely (Rayleigh quotient), or all at once (QR algorithm).

**What's next and why it matters:** Now we leave the world of matrices and enter the world of waves and frequencies — the Fast Fourier Transform is the machine that decomposes any signal into its constituent notes, and it's behind everything from music compression to solving PDEs.
