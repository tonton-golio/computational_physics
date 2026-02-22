# Linear Equations

> *Remember the [condition number](./bounding-errors)? Watch how it shows up again here — this time deciding whether your linear system's answer is trustworthy or garbage.*

## Big Ideas

* In finite dimensions, every linear map *is* a matrix — the abstract and the concrete are secretly the same thing once you pick a basis.
* LU decomposition works by recording the steps of Gaussian elimination; solving then costs almost nothing compared to the factorization itself.
* Never compute $A^{-1}$ explicitly — it amplifies rounding errors. Factorize instead, and solve by substitution.
* The condition number of the matrix is the multiplier that turns input uncertainty into output uncertainty; a large condition number means the answer is geometrically fragile.

## Why Linear Systems Matter

Linear systems are special: they're something we can solve really well, and they cover an enormous range of problems. Even when you've got a nonlinear system, you can isolate the linearity and solve that, treating the nonlinearity separately.

## From Abstract Linear Systems to Matrices

A **linear function** is anything that behaves linearly. A mapping $F: X \rightarrow Y$ is linear if $F(ax + bx') = aF(x) + bF(x')$.

*Linearity means "scaling and adding inputs does the same thing as scaling and adding outputs." It's the nicest property a function can have.*

Linear functions and matrices are the same thing (in finite dimensions, once bases are chosen):
* Pick a basis $\{e_{1}, e_2, \dots e_n\}$ for $X$, so any $x \in X$ can be written uniquely as $x = a_1e_1 + \dots + a_n e_n$
* Pick a basis $\{f_{1}, f_2, \dots f_m\}$ for $Y$
* Then $F(e_j)=b_{1j}f_1 + \dots + b_{mj}f_m$
* Expand everything out:
	$$ F(x) = \sum_{i=1}^m f_i \left(\sum_{j=1}^n a_j b_{ij} \right) $$
	That thing on the right is just a matrix product.

The matrix entries? $b_{ij} = \left< f_i \vert F e_j \right>$.

**Punch Line:** The actions of $F: X\rightarrow Y$ are exactly the same as $F: \mathbb{C}^m \rightarrow \mathbb{C}^n$.

### How to solve linear systems

Find all $x \in X$ such that $F(x) = y$:
1. Pick basis sets for $X$ and $Y$
2. Write down matrix $F$ and RHS $y$ in those bases
3. Solve the matrix equation $Fx=y$
4. Dot the result with the $e$ basis to get back to the problem domain

## Existence of Solutions

Today we use the case where $m=n$ (square matrix). Three cases:

* **$F$ is non-singular**: $Im(F) = Y$, $Rank(\underline{F}) = n$, $det(\underline{F}) \neq 0$, trivial kernel, unique solution exists
* **$F$ is singular**, $y \not\in Im(F)$: no solutions
* **$F$ is singular**, $y \in Im(F)$: infinitely many solutions (add anything from the kernel)

Now the real question: even when exact solutions exist, are they *stable*?

## Condition Number Interactive Demo

[[simulation condition-number-demo]]

## Sensitivity of a Linear System

The more orthogonal the matrix rows, the lower the condition number. Nearly parallel rows? The condition number explodes.

Picture two lines crossing at a sharp angle — the intersection is well-defined, even if the lines wiggle. Now picture two nearly parallel lines — a tiny wiggle moves the intersection wildly. That's what a high condition number looks like in 2D.

## Interactive Gaussian Elimination and LU Decomposition

[[simulation gaussian-elim-demo]]

[[simulation lu-decomp-demo]]

## Building the Algorithms

We want to transform $Ax = b$ into $x$ using operations that are linear, invertible, and cheap.

**Fundamental Property:** if $M$ is invertible, then $MAx = Mb$ has the same solutions as $Ax=b$. Multiplying both sides by the same invertible matrix doesn't change the answer — like weighing something on a different scale.

Why not just compute $A^{-1}$ directly? Because computing an inverse is like photocopying a photocopy — each generation loses quality. LU factorization avoids this by never forming the inverse explicitly.

The strategy:
1. Factor $A = LU$ (lower and upper triangular)
2. Each step, zero out one column below the diagonal using row scaling and subtraction
3. Both operations are linear and invertible

## lu_factorize

Watch this little magician at work:

```python
def lu_factorize(M):
    """
    Factorize M such that M = LU
    Split the matrix into a lower triangle and an upper triangle.
    """
    m, U = M.shape[0], M.copy()
    L = np.eye(M.shape[0])  # start with identity
    for j in range(m-1):
        if U[j, j] == 0:
            raise ValueError(f"Zero pivot at position ({j},{j}). Use partial pivoting for stability.")
        for i in range(j+1, m):
            scalar    = U[i, j] / U[j, j]  # how much of row j to subtract
            U[i]     -= scalar * U[j]       # eliminate the entry below the pivot
            L[i, j]   = scalar              # remember the multiplier for later

    return L, U
```

## forward_substitute

**Back-substitution is like unwrapping a present.** Once you have $LU$, solving $Ly = b$ is like opening boxes from the outside in. The first equation gives you $y_1$ immediately. Plug that into the second and you get $y_2$. Each step peels off one more layer.

```python
def forward_substitute(L, b):
    """
    Solve Ly = b by peeling off one layer at a time.
    """
    y = np.zeros(np.shape(b))
    for i in range(len(b)):
        y[i] = (b[i] - L[i] @ y) / L[i, i]  # peel off the next layer
    return y
```

## backward_substitute

Then $Ux = y$ goes the other direction — start from the last equation and work backwards:

```python
def backward_substitute(U, y):
    """
    Solve Ux = y by working from the bottom up.
    """
    x = np.zeros(np.shape(y))
    for i in range(1, 1 + len(x)):
        x[-i] = (y[-i] - U[-i] @ x) / U[-i, -i]  # unwrap from the inside out
    return x
```

## solve_lin_eq

```python
def solve_lin_eq(M, z):
    """The full solve: factorize, then unwrap twice."""
    L, U = lu_factorize(M)
    y = forward_substitute(L, z)
    return backward_substitute(U, y)
```

> **Challenge.** Build a 5x5 random matrix in NumPy. Solve $Ax = b$ using your own `solve_lin_eq` and compare with `np.linalg.solve`. How many digits agree? Now make the matrix nearly singular (e.g., make two rows almost identical) and try again. Watch the digits evaporate.

## Conditioning and Error Analysis

Consider $Ax=b$. Each equation $a_{i1}x_1 + a_{i2}x_2 + \dots + a_{in}x_n = b_i$ defines a hyperplane.

In 2D, two equations define two lines. The intersection is the solution. Now add uncertainty: $||\Delta b||_\infty < \delta$. Those lines become ribbons, and the solution becomes a region. Nearly parallel lines? That region is enormous. That's ill-conditioning, visible right there in the geometry.

### Error bounds:

$A(x+\Delta x) = (b+\Delta b)$ implies:

$$\frac{||\Delta x||}{||\hat{x}||} \leq \text{COND}(A) \frac{||\Delta b||}{||\hat{b}||}$$

*The relative error in your solution is at most COND(A) times the relative error in your data. If COND(A) = 1000 and your data has 0.1% error, your solution could have up to 100% error.*

So how do you know if your answer is any good? Compute `np.linalg.cond(A)`. If it's near 1, relax. If it's $10^{10}$, panic.

---

## What Comes Next

Linear systems are paradise: existence and uniqueness are decided by a single rank check, and LU reaches the exact answer in a fixed number of steps. But real data is noisy, and real experiments give you *more* equations than unknowns — that's the least-squares problem, and it's next.

## Check Your Understanding

1. You solve $Ax = b$ using LU decomposition and get a residual $\|b - A\hat{x}\|$ that is nearly zero, yet the true error $\|x - \hat{x}\|$ is large. What property of $A$ makes this possible?
2. Forward substitution solves $Ly = b$ top to bottom; backward substitution solves $Ux = y$ bottom to top. Why does the triangular structure make each step trivial?
3. Two nearly parallel lines in 2D give a large condition number. Explain this geometrically.

## Challenge

Construct a family of $n \times n$ matrices parameterized by an angle $\theta$ whose rows are unit vectors that all point nearly in the same direction (angle $\theta$ between adjacent rows). Compute the condition number and the relative error in $x$ as a function of $\theta$ for $\theta \in [0.01°, 90°]$. At what angle does the system become effectively unsolvable in double precision?
