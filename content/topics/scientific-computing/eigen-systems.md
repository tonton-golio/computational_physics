# Eigenvalue Problems

> *Every matrix has a set of special directions that it only stretches — never rotates. These are the eigenvectors, and the stretch factors are the eigenvalues. Together, they reveal the DNA of the matrix.*

## Big Ideas

* An eigenvector is a direction the matrix refuses to rotate — it only stretches. Every stable physical system whispers its eigenvectors to you through its natural modes.
* The power method is the laziest eigenvalue finder: multiply and normalize until the biggest voice drowns out the rest. Convergence rate = ratio of the two largest eigenvalues.
* Rayleigh quotient iteration achieves cubic convergence through a virtuous feedback loop: better eigenvector gives a better shift gives a better eigenvector...
* The QR algorithm finds *all* eigenvalues simultaneously via repeated similarity transformations that preserve the spectrum while driving off-diagonal entries to zero.

Any stable physical or chemical system can be described with an eigensystem. Wavefunctions in quantum mechanics are a typical example.

$$
Ax = \lambda x
$$

Where $A$ is the operator, $x$ an eigenvector, and $\lambda$ an eigenvalue. The question: **How do we find them?**

[[simulation eigen-transformation]]

## Why Eigenvalues Matter

* **Quantum mechanics**: Energy levels are eigenvalues of the Hamiltonian
* **Vibrations**: Natural frequencies of structures
* **Stability analysis**: Behavior near equilibrium
* **PCA**: Dimensionality reduction in data science
* **PageRank**: Largest eigenvector of the web graph

*Every time you ask "what are the natural modes of this system?" you're asking an eigenvalue question.*

---

## Mathematical Foundations

### The Characteristic Polynomial

$(A-\lambda I)x = 0$ has non-trivial solutions iff $\det(A-\lambda I) = 0$.

*Eigenvalues are values of $\lambda$ that make $(A - \lambda I)$ singular.*

[[simulation characteristic-polynomial]]

### Eigenspaces and Multiplicity

The characteristic polynomial has degree $n$, so exactly $n$ complex roots (counting multiplicity). Two types of multiplicity matter:
1. **Algebraic**: How many times $\lambda_i$ appears as a root
2. **Geometric**: Dimension of the eigenspace $E_\lambda$

Key: geometric $\leq$ algebraic.

### Defective vs Non-Defective

* **Non-defective**: complete eigenbasis exists. You can diagonalize: $A = T\Lambda T^{-1}$. Computing $A^{100}$? Just compute $\lambda_i^{100}$. Beautiful.
* **Defective**: not enough eigenvectors. Life is harder.

Non-defectiveness is guaranteed if $A$ is normal ($A^H A = AA^H$), or all eigenvalues are distinct, or $A$ is Hermitian ($A=A^H$, best case — orthonormal eigenbasis).

[[simulation hermitian-demo]]

---

## Part I: Simple Methods (one eigenvalue at a time)

### The Power Method — the laziest eigenvalue finder

Take any vector and multiply it by $A$ over and over. The dominant eigenvalue's eigenvector gradually takes over, like the loudest voice drowning out everyone else.

Order eigenvalues by magnitude: $|\lambda_1| \geq |\lambda_2| \geq \dots$

After $k$ multiplications:
$$A^k x = \lambda_1^k \left(\tilde{x}_1 t_1 + \tilde{x}_2 \left(\frac{\lambda_2}{\lambda_1}\right)^k t_2 + \dots\right)$$

The ratios $(\lambda_i/\lambda_1)^k \to 0$, so only the dominant eigenvector survives. Convergence rate: $O(|\lambda_2/\lambda_1|^k)$.

[[simulation power-method-animation]]

```python
def power_iterate(A, x0, tol=1e-10, max_iter=1000):
    """
    The laziest eigenvalue finder: just keep multiplying and normalizing.
    """
    x = x0 / np.linalg.norm(x0)

    for i in range(max_iter):
        y = A @ x                       # multiply by A
        x_new = y / np.linalg.norm(y)   # normalize so it doesn't blow up

        eigenvalue = x_new.conj() @ A @ x_new  # Rayleigh quotient

        if np.linalg.norm(x_new - x) < tol or np.linalg.norm(x_new + x) < tol:
            return eigenvalue, x_new, i + 1

        x = x_new

    return eigenvalue, x, max_iter
```

**Limitations:** Only finds the dominant eigenvalue. Fails if it's not unique. Slow when $|\lambda_1| \approx |\lambda_2|$.

---

### Inverse Iteration — finding eigenvalues near a target

Apply the power method to $(A - \sigma I)^{-1}$. Its eigenvalues are $(\lambda_i - \sigma)^{-1}$, so the eigenvalue closest to $\sigma$ becomes dominant.

*Flip the spectrum inside-out. The one you care about becomes the biggest.*

```python
def inverse_iterate(A, sigma, x0, tol=1e-10, max_iter=1000):
    """
    Flip the spectrum so the one you want becomes dominant.
    """
    n = len(A)
    x = x0 / np.linalg.norm(x0)
    A_shifted = A - sigma * np.eye(n)

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

The **Rayleigh quotient** $\lambda_R(x) = \frac{x^T A x}{x^T x}$ is the optimal eigenvalue estimate for a given direction $x$.

*Project $A$ onto the direction of $x$ to get your best guess for the eigenvalue.*

Update the shift at each step and watch the magic: **cubic convergence** for Hermitian matrices.

Here's the feedback loop that makes it happen. The Rayleigh quotient error is quadratic in the eigenvector error: $|\lambda_R - \lambda^*| = O(\|x - x^*\|^2)$. So: (1) better eigenvector gives quadratically better shift, (2) better shift makes inverse iteration converge faster, (3) faster convergence gives even better eigenvector. Composing these gains yields cubic convergence. Explosive.

```python
def rayleigh_quotient_iteration(A, x0, tol=1e-12, max_iter=100):
    """
    Cubic convergence — the shift chases the eigenvalue in a virtuous spiral.
    """
    n = len(A)
    x = x0 / np.linalg.norm(x0)
    eigenvalue = x @ A @ x

    for i in range(max_iter):
        A_shifted = A - eigenvalue * np.eye(n)
        try:
            y = np.linalg.solve(A_shifted, x)
        except np.linalg.LinAlgError:
            return eigenvalue, x, i    # singular = we nailed it

        x_new = y / np.linalg.norm(y)
        eigenvalue_new = x_new @ A @ x_new

        if abs(eigenvalue_new - eigenvalue) < tol:
            return eigenvalue_new, x_new, i + 1

        x = x_new
        eigenvalue = eigenvalue_new

    return eigenvalue, x, max_iter
```

[[simulation rayleigh-convergence]]

| Method | Convergence | Finds |
|--------|------------|-------|
| Power Method | Linear | Dominant $\lambda$ |
| Inverse Iteration | Linear | $\lambda$ near shift |
| Rayleigh Quotient | **Cubic** (Hermitian) | Single $\lambda$ |

---

## Part II: Fancy Methods (all eigenvalues at once)

### QR Algorithm

The industry workhorse — the method behind `numpy.linalg.eig`.

$$A_k = Q_k R_k, \qquad A_{k+1} = R_k Q_k$$

Each iteration is a similarity transformation: $A_{k+1} = Q_k^T A_k Q_k$. Same eigenvalues, but the off-diagonal entries gradually shrink to zero. When they vanish, eigenvalues appear on the diagonal.

[[simulation qr-algorithm-animation]]

```python
def qr_algorithm(A, max_iter=1000, tol=1e-12):
    """
    Factorize, reverse-multiply, repeat until eigenvalues appear on the diagonal.
    """
    A_k = A.copy()
    for i in range(max_iter):
        Q, R = np.linalg.qr(A_k)
        A_k = R @ Q
        off_diag = np.sum(np.abs(np.tril(A_k, -1)))
        if off_diag < tol:
            break
    return np.diag(A_k), A_k, i + 1
```

**Practical speedups:** Hessenberg reduction first, Wilkinson shifts, implicit QR.

---

### Functions of Matrices

For non-defective $A = T\Lambda T^{-1}$:

$$f(A) = T f(\Lambda) T^{-1}$$

*To apply a function to a matrix, diagonalize it, apply the function to each eigenvalue separately, and transform back. Want $e^A$? Compute $e^{\lambda_i}$ for each eigenvalue.*

The matrix exponential $e^{At}$ solves $\frac{dx}{dt} = Ax$ — connecting eigenvalues directly to dynamical systems and [initial value problems](./initial-value-problems).

[[simulation matrix-exponential]]

---

> **Challenge.** Make a random 4x4 matrix, compute its Gershgorin circles (center = diagonal entry, radius = sum of absolute off-diagonal entries in that row), and plot them. Then plot the actual eigenvalues as dots. Are they all inside the circles? (They must be!)

[[simulation gershgorin-circles]]

## What Comes Next

Eigenvalues are the frequency domain of matrices: they tell you the natural scales of a system. The Fast Fourier Transform is the frequency domain of signals. The two ideas are deeply connected — and the FFT is next.

## Check Your Understanding

1. Describe a simple modification of the power method that finds the eigenvalue *closest to a given target* $\sigma$.
2. A matrix has $\lambda_1 = 3$ and $\lambda_2 = -3$. What happens when you run the power method, and why?
3. Show that $A_{k+1} = R_k Q_k$ is similar to $A_k$ (same eigenvalues).

## Challenge

Construct a symmetric $5 \times 5$ matrix $A = Q \Lambda Q^T$ with $\Lambda = \text{diag}(1, 2, 5, 10, 50)$ and $Q$ a random orthogonal matrix. Run power iteration, inverse iteration (targeting $\lambda \approx 5$), and Rayleigh quotient iteration. Track $|\lambda_k - \lambda^*|$ at each step and plot all three on a log scale. How many steps does each need to reach machine precision?
