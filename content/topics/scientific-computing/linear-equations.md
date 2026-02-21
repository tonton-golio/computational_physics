# Linear Equations
*The one family of problems we can solve perfectly (almost)*

> *Remember the condition number from Lesson 01? Watch how it shows up again here — this time deciding whether your linear system's answer is trustworthy or garbage.*

## Why Linear Systems Matter
Linear systems are special: they're something we can solve really well, and they cover an enormous amount of problems. Even when you've got a non-linear system, you can isolate the linearity and solve that, and treat the non-linearity in another way.

## From Abstract Linear Systems to Matrices
#### Going from abstract linear systems to matrices

A **linear function** is anything that behaves in the way we call linear. They exist in vector spaces: A mapping $F: X \rightarrow Y$ (abstract, eg: wave functions) is linear if $F(ax + bx') = aF(x) + bF(x')$.

*This says: linearity means "scaling and adding inputs does the same thing as scaling and adding outputs." It's the nicest property a function can have.*

Linear functions and matrices are the same thing (in finite dimensions, once bases are chosen)!
* Let's say you have a basis for $X$, eg: $\{e_{1}, e_2, \dots e_n\}$, then any vector $x \in X$ can be written uniquely as $x = a_1e_1 + \dots + a_n e_n$
* Let $\{f_{1}, f_2, \dots f_n\}$ form a basis for $Y$, any vector $y \in Y$ can be written as $y = b_1f_1 + \dots + b_m f_m$
* $$F(e_j)=b_{1j}f_1 + \dots + b_{mj}f_m$$
	We can write it because of linearity, but you'll notice that it's just matrix multiplication.
* $$ F(x) = F(a_1e_1 + \dots + a_ne_n) = a_1F(e_1) + \dots + a_nF(e_n) $$
	$$ = \sum_{j=1}^n a_j \sum_{i=1}^m b_{ij} f_i $$
	$$ = \sum_{i=1}^m f_i \left(\sum_{j=1}^n a_j b_{ij} \right) $$
	The thing on the right is nothing more than a matrix product. Think of the total thing as:  Representation of $F$ in $e,f$ basis $*$ (Coordinates of $x$ in $e$ basis) = Coordinates of $y$ in $f$ basis

We've shown how an abstract linear problem can be represented with matrices.

We're missing one thing: how do you find $b_{ij}$, the components of the matrix?
$$b_{ij} = \left< f_i \vert F e_j \right>$$
For example, $\int f_i(\alpha)(Fe_j)(\alpha) d\alpha$, if $Y$ is a space of functions $f(\alpha)$

**Punch Line:** The actions of $F: X\rightarrow Y$ are exactly the same as $F: \mathbb{C}^m \rightarrow \mathbb{C}^n$.

#### How to solve linear systems of equations

Find all $x \in X$ such that $F(x) = y$ (some known $y\in Y$)
1. Pick basis sets $\{e_1, \dots, e_n\}$ for $X$, $\{f_1, \dots, f_m\}$ for $Y$
2. Write down matrix $F$ in $e,f$ basis and RHS $y$ in $f$ basis
3. Solve matrix equation $Fx=y$, where $F$ and $y$ are known
4. Dot result $x$ with $e$ basis to get the result in the problem domain $x=x_1e_1+\dots+x_ne_n$

## Existence of Solutions
#### Can we solve $F(x)=y$?
Yes, when solutions exist!

Today, we'll use the case where $m=n$, i.e, matrix $\underline{F}$ is square.

If you apply $F$ to all the vectors in the source space, you get a subset (not proper) of $Y$, which is called the _image_. $Im(F) = \{F(x) \vert x \in X\}$.

In the non-abstract space, it's called the _span_: $Span(\underline{F}) = \{\underline{F} \,\underline{x} \vert \underline{x} \in \mathbb{C}\}$

Three cases when $n=m$

* **$F$ is non-singular**: (if one of them holds, all of them hold: they're equivalent)
* $Im(F) = Y$ (Dimension of source space $X$ = Dimension of target space $Y$)
* $Span(\underline{F} = \mathbb{C}^n)$
* $Rank(\underline{F}) = n$
* $det(\underline{F}) \neq 0$
* $\underline{F}\,\underline{x} = 0 \leftrightarrow \underline{x}=0$ (trivial kernel)
* $\underline{F}$ is invertible (Deterministically find unique solution)
* **$F$ is singular**:
* $Im(F) \not\subseteq Y$
* $Span(\underline{F} \not\subseteq \mathbb{C}^n)$
* $Rank(\underline{F}) < n$
* $det(\underline{F}) = 0$
* There is a non-trivial subspace $Ker(\underline{F})$ such that $\underline{F}(\underline{x})=0$ for $\underline{x}\in Ker(\underline{F})$

Singular $F$ splits into two sub-cases:
1. $y \not\in Im(F) \implies$ there are no solutions
2. $y \in Im(F) \implies$ infinitely many solutions (because we can add something from the kernel and get another solution).

That was the mathematical part: Now we're going to look at a case where we have exact solutions: when are the solutions stable, and when do small perturbations cause it to blow up?

## Condition Number Interactive Demo

<ConditionNumberDemo />

## Sensitivity of a Linear System of Equations

The more orthogonal the matrix is, the lower the condition number. It's ~1 for orthogonal, but as they get closer to one another, i.e, they get closer to being linearly dependent on one another, condition number increases. The webpage has a calculation of the exact condition number.

## Interactive Gaussian Elimination and LU Decomposition

<GaussianElimDemo />

<LUDecompDemo />

## How to build the algorithms from scratch

Construct algorithms that transform $b=Ax$ into $x$ using a modest number of operations that are linear, invertible, and simple to compute.

**Fundamental Property:** if $M$ is invertible, then $MAx = Mb$ has the same solutions as $Ax=b$

*This says: multiplying both sides of the equation by the same invertible matrix doesn't change the answer. It's like weighing something on a different scale — the object doesn't change.*

What we know:
1. Our algorithm must have the same effect as multiplying by the inverse. We avoid explicitly computing $A^{-1}$ and multiplying because inverses amplify rounding errors, especially for ill-conditioned matrices: $A \rightarrowtail I; I \rightarrowtail A; b \rightarrowtail x$
2. Each step must be linear, invertible
3. We first want $A = LU$ (lower and upper triangular matrices)
4. Every step, we want to take an $n\times n$ matrix, and reduce the leftmost column to be zero below the first element. Then, we just recursively continue for an $(n-1)\times(n-1)$ matrix
5. Row scaling (it's linear and invertible)
6. Row addition and subtraction is also linear and invertible

We can use 5. and 6. together: just scale the rows (if the leading term is nonzero) and subtract

> **You might be wondering...** "Why not just compute $A^{-1}$ directly?" Because computing an inverse is like photocopying a photocopy — each generation loses quality. With floating-point arithmetic, the rounding errors in computing $A^{-1}$ get amplified when you multiply by it. LU factorization avoids this by never forming the inverse explicitly.

## lu_factorize
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

**Back-substitution is like unwrapping a present.** Once you have $LU$, solving $Ly = b$ (forward substitution) is like opening boxes from the outside in. The first equation gives you $y_1$ immediately (one unknown, one equation). Plug that into the second equation and you get $y_2$. Each step peels off one more layer of wrapping until the whole present is unwrapped.

```python
def forward_substitute(L, b):
    """
    Solve Ly = b by peeling off one layer at a time.
    The first equation has only y[0], the second has y[0] and y[1], etc.
    """
    y = np.zeros(np.shape(b))
    for i in range(len(b)):
        y[i] = (b[i] - L[i] @ y) / L[i, i]  # peel off the next layer
    return y
```

## backward_substitute

Then $Ux = y$ (backward substitution) goes the other direction — start from the last equation (one unknown) and work backwards, unwrapping from the inside out.

```python
def backward_substitute(U, y):
    """
    Solve Ux = y by working from the bottom up.
    The last equation has only x[n-1], the second-to-last has x[n-1] and x[n-2], etc.
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

> **Challenge:** Build a 5x5 random matrix in NumPy. Solve $Ax = b$ using your own `solve_lin_eq` and compare with `np.linalg.solve`. How many digits agree? Now make the matrix nearly singular (e.g., make two rows almost identical) and try again. Watch the digits evaporate.

## Conditioning and Error Analysis
Consider the matrix equation $Ax=b$.
For $i = 1, \dots, m$: $a_{i1}x_1 + a_{i2}x_2 + \dots + a_{in}x_n \equiv A_i^T x=b_i$. Each equation defines a hyperplane in $\mathbb{R}^n$.

If there are 2 unknowns, i.e., $n=2$:
$$a_{11}x_1 + a_{12} x_2 = b_1$$
$$a_{21}x_1 + a_{22} x_2 = b_2$$
In two dimensions, these define lines, we can find the slope and y-intercepts. The point of intersection is the solution to the system.

Assume that there's an error or uncertainty in the data, so that $||\Delta b||_\infty < \delta$. You'll have a region of uncertainity around the lines, transforming them into rectangles. Now, your solution can be anywhere in the shape that's made from the intersection of the rectangles. If we had an error in the matrix, we should have a skew (and a shift in slopes) and not a vertical shift like we do if the error is in $b$.

> **You might be wondering...** "What does this look like geometrically?" Picture two lines crossing at a sharp angle — the intersection point is well-defined, even if the lines wiggle a bit. Now picture two lines that are almost parallel — a tiny wiggle moves the intersection point wildly. That's what a high condition number looks like in 2D.

#### Error bounds:

1. RHS: Assume $A \hat{x} = \hat{b} \implies A(x+\Delta x) = (b+\Delta b)$. That implies:
	* $||\hat{b}|| = ||A\hat{x}|| \leq ||A||\; ||\hat{x} ||$ and so $||\hat{x}|| \geq \frac{||\hat{b}||}{||A||}$
	* $|| \Delta x || = || A^{-1} \Delta b || \leq || A^{-1}|| \; ||\Delta b ||$
	* together, we get $\frac{||\Delta x||}{||\hat{x}||} \leq ||A^{-1}|| \; ||A|| \; \frac{||\Delta b||}{||\hat{b}||} = \text{COND}(A) \frac{||\Delta b||}{||\hat{b}||}$

*This says: the relative error in your solution is at most COND(A) times the relative error in your data. If COND(A) = 1000 and your data has 0.1% error, your solution could have up to 100% error.*

2. Nothing here depends on matrix $A$ being exact, and we can replace $A$ here with $\hat{A}$ everywhere so we can get a calculation even if it's not exact

> **You might be wondering...** "So how do I know if my answer is any good?" Compute the condition number! In NumPy: `np.linalg.cond(A)`. If it's near 1, relax. If it's $10^{10}$, panic (or at least be very suspicious of your answer).

---

**What we just learned in one sentence:** We can solve linear systems exactly by splitting the matrix into triangles (LU), but the condition number decides how much we should trust the answer.

**What's next and why it matters:** Now that we can solve linear systems perfectly, imagine the data itself is noisy and there are more equations than unknowns — that's where least squares comes in, and it's the same trick astronomers use to find planets from messy telescope data.
