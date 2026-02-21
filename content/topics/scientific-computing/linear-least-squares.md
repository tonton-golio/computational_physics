# Linear Least Squares

> *Remember the [condition number](./bounding-errors)? Here's where it bites us — the normal equations square it, and Householder reflections save us.*

## Overdetermined Systems: m > n

Large number of equations, small number of unknowns. In general, there are _no exact solutions_.

All the places you can reach are driven by $Im(A)$, the image. There's no $x$ that you can feed in that results in you ending up outside the image.

### How would a least squares approximate solution have to look?

**Write** $\mathbb{C}^m = Im(A) \oplus Im(A)^\perp$
Any vector space with a subspace can be broken down into the subspace and a space that's orthogonal to the subspace.
**Means** $b \in \mathbb{C}^m$ can be uniquely written $b = \tilde{b} + b^\perp$, with $\tilde{b} \in Im(A), b^\perp \in Im(A)^\perp$
**Where**
$$Im(A)^\perp = \{x\in \mathbb{C}^m \mid x^T x' = 0 \;\forall\; x' \in Im(A)\}$$

$$=\{x\in \mathbb{C}^m \mid A^T_i x=0 \text{ for } 1 \leq i \leq m \}$$

Watch this trick — Pythagoras makes the whole thing click:

$$||r||^2 = ||b-Ax||^2$$
$$= ||\tilde{b} + b^\perp - Ax||^2 = ||(\tilde{b}-Ax)+b^\perp||^2$$
Since $(\tilde{b}-Ax) \in Im(A)$ and $b^\perp \in Im(A)^\perp$, these two components are orthogonal. By Pythagoras (valid in $L^2$ norm because orthogonal vectors satisfy $\|u+v\|^2 = \|u\|^2 + \|v\|^2$):
$$ = ||\tilde{b}-Ax||^2 + || b^\perp ||^2$$

*This says: the total error splits into a part we can control (by choosing x) and a part we can't (the component of b perpendicular to the image). Minimize the first, accept the second.*

The first term is in the image, the second is in the perpendicular subspace. We can minimize the first term to zero (by choosing $Ax = \tilde{b}$), and the second term is a constant independent of $x$.

**So:** the residual of least squares is perpendicular to $Im(A)$.

## The Least Squares Problem
With given data and a desired function, determine the parameters of the function to minimize the distance to data points.


## Least Squares Solution
* **Always Exists:** the solution to $Ax=\tilde{b}$ is the least squares solution.
$\tilde{b}$ is the projection on the image space, and $b^\perp$
the projection on the orthogonal component.
* **Unique if Rank(A) = n:** If the rank is less than $n$ (columns are linearly dependent), there are
infinitely many solutions: you can find one and add the kernel space
to it to get them all.
* **Normal equations**:
$A^T_i r=0 \Rightarrow A^T r=0 \implies A^T(b-Ax)=A^Tb-A^TAx=0$.
The fact that the residual is orthogonal to the image means that
for all the rows of $A$, the dot product with the residual has to be 0.
* **$A^TA$ is a small square matrix**
* The only problem is that this has a large condition number: $\text{COND}(A^TA) = \text{COND}(A)^2$

*This says: forming the normal equations squares the condition number. If your matrix was a bit ill-conditioned (say COND = 1000), the normal equations make it horrifically ill-conditioned (COND = 1,000,000). That's like photocopying a blurry photo — every generation makes it worse.*

We're almost at our goal, but not there yet. We've found an efficient solution (the normal equations) which are great for mathematical calculations, but we can't use them because a small error will ruin us because of the squared condition number.

### How do we save our significant digits (hopefully without too much work?)

Note that $Im(A)^\perp = Ker(A^T) \implies \mathbb{C}^m = Im(A) \oplus Ker(A^T)$.

### Recap
We decomposed the target space $\mathbb{C}^m$ into the image of $A$ and the kernel of $A^T$, two orthogonal subspaces.

$\text{COND}(A^TA) = \text{COND}(A)^2$, but $\text{COND}(Q^TA) = \text{COND}(A)$.

We just need to construct the effect of multiplying our matrix by $Q^T$. In general, you never want to calculate $Q$ because it can be really large (million x million).

Why does multiplying by $Q^T$ not mess up the condition number, but multiplying by $A^T$ does? Because $Q$ is orthogonal — it's a pure rotation/reflection that doesn't stretch anything. Multiplying by $Q^T$ is like turning your head to look at the problem from a different angle. The problem doesn't change, just your viewpoint. Multiplying by $A^T$ is like squishing the problem through a funnel.

## Building a least squares algorithm from scratch

Note: Norm $\left(\,||x||\,\right)$ always refers to the Euclidean Norm $\left(\,||x||_2\,\right)$ when we're talking about least squares.

**Goal:** Construct the effect of $Q^T$ such that $Q^T A = R$, where $R$ is a matrix with the bottom part being 0 and the upper part being upper triangular.

### The Householder reflection: bouncing light off a mirror

Here's the beautiful part. Imagine you're holding a flashlight (that's your column vector $a$) and you want to redirect all the light so it points straight along the first axis $e_1$ (that's the target direction). How do you do it? **Place a mirror at exactly the right angle between the current direction and the target direction, and bounce the light off it.** That's exactly what a Householder reflection does — it's a mirror that zeros out everything below the diagonal in one bounce.

**Building blocks:** We want to perform _unitary_ operations: i.e., do things that don't change the length of vectors: operations like rotation, reflection (not translations, because although translations don't change lengths, they change the norm).
* 2D: Rotations and reflections
* 3D: Rotations, reflections and inversions (like reflection through a point)
* Higher dimensions: lots more (symmetries of n-spheres), permutations (everywhere)

**What do we want to build?** Similar to LU, we want to take a column and eliminate everything below the diagonal. Construct a reflection operation $H$ (for Householder), which is a reflection. The norm of the entire column must be the same though, and so the top element (the diagonal) must contain $||a||$ concentrated in it.

**Step 1: Set up the mirror.** Reflect vector (column) $a$ onto basis vector $e_1$. Consider a mirror, which is the angle bisector in between $a$ and $e_1$. We could reflect it onto $e_1$ or even onto the negative side, $-e_1$. Call the two mirrors $v^+$ and $v^-$.

**Step 2: Bounce the light.** The operation needs to transform $\vec{a} \to \alpha \vec{e_1}$, where $\alpha= \pm ||a||$. For numerical stability, choose $\alpha = -\text{sign}(a_1) \|a\|$ to avoid catastrophic cancellation.

$$Ha = a - 2 P_v a$$
$$ H = I - 2P_v \qquad v=v^+ \text{ or } v^-$$

**Step 3: Find the mirror's orientation.** How to find $v^+, v^-$?

Projection operator, $P_v a = \frac{v^T a}{||v||^2} v= \frac{v^T a}{v^T v} v = \left(\frac{vv^T}{v^T v}\right) a$
$$\implies P_v = \frac{vv^T}{v^Tv}$$

$$\alpha e_1 = Ha = a-2\frac{v^Ta}{v^Tv}v$$
$$\underbrace{\frac{2v^Ta}{v^Tv}}_{\text{scalar}} v = a-\alpha e_1$$
First part is just a scalar number, the rest are vectors. It gets normalized away if we choose $||v||=1$.

Thus, $v \propto a - \alpha e_1$. Just subtract $\alpha$ from the first entry in the column.

Now we can plug that into the equation for $H$, and we're done.

Why not just use rotations like Givens rotations? You can! But Householder zeros out an entire column in one shot (one mirror bounce), while Givens rotations zero out one entry at a time (many small rotations). Householder is the sledgehammer; Givens is the scalpel. Use the sledgehammer for dense matrices, the scalpel for sparse ones.

> **Challenge.** Take a random 3D vector and compute the Householder reflection vector $v$. Apply $H = I - 2vv^T/(v^Tv)$ and verify the result points along $e_1$ with the same norm. Do it in 5 lines of NumPy.

## Code
```python
def HouseholderQR(A, b):
	m, n = A.shape
	R = A.astype(float)
	b_tilde = b.astype(float)
	for k in range(n):
		a = R[k:m, k]
		if (np.dot(a[1:], a[1:]) == 0):
			continue                        # already zeroed out, skip
		v = reflection_vector(a)            # find the mirror orientation
		reflect_columns(v, R[k:m, k:n])    # bounce the matrix off the mirror
		reflect_columns(v, b_tilde[k:m])   # bounce b too
	return R, b_tilde

def reflect_columns(v, A):
	"""Apply Householder reflection defined by v to columns of A.
	Computes A = (I - 2 vv^T / (v^T v)) A in-place — like a mirror reflection, zap!"""
	S = -2 * np.dot(v, A) / np.dot(v, v)
	A += v[:, np.newaxis] * S[np.newaxis, :]
```

## QR Factorization via Householder Reflections
The last line RHS is equivalent to `np.outer(v, S)`.

This then yields a much better least squares optimizer:


## least_squares
```python
def least_squares(A, b):
    """
    Solve min ||Ax - b||_2 for x using QR factorization.
    Project b onto the column space of A without squaring the condition number.
    """
    Q, R = householder_QR_slow(A)
    v = Q.T @ b                        # rotate b into Q's frame
    size = np.shape(A)[1]
    R_cropped = R[:size, :size]         # keep the upper triangular part
    v_cropped = v[:size]                # keep the part in the column space
    x = backward_substitute(R_cropped, v_cropped)  # unwrap the present

    # The residual norm: entries of Q^T b below the first 'size' rows
    # are the component of b orthogonal to Im(A) — the part we can't fix
    residual = np.linalg.norm(v[size:])
    return x, residual
```

> **Challenge.** Generate 20 noisy data points from $y = 2x + 3 + \text{noise}$. Fit a line using your `least_squares` function and compare with `np.polyfit`. Plot both. They should agree — but now you understand what's happening under the hood.

---

## Big Ideas

* When there are more equations than unknowns, the residual inevitably has a component perpendicular to the image of $A$ — that part is irreducible, no matter how clever your solver.
* Forming the normal equations $A^T A x = A^T b$ squares the condition number, which is like photocopying a blurry photo — every generation loses quality.
* Householder reflections are orthogonal transformations (pure rotations and reflections), so they never stretch the problem and never amplify the condition number.
* QR factorization is the right algorithm not because it is elegant (though it is), but because it is the numerically stable way to get what the normal equations give you algebraically.

## What Comes Next

Least squares is the last linear paradise. From here on, the world is nonlinear: equations where you cannot express the solution as a matrix times a vector, where the number of solutions is unknown, and where convergence is a gift rather than a guarantee.

The good news is that nonlinear methods almost always work by *linearizing* at each step — replacing the true problem with a locally linear one and solving that. The tools you built here — LU for the linear solve inside each Newton step, the condition number as a warning light, and the geometric intuition about projections — all carry forward directly into the nonlinear world.

## Check Your Understanding

1. You have 100 measurements of a physical quantity and you want to fit a polynomial of degree 3. How many columns does the matrix $A$ have, how many rows, and why is the system overdetermined?
2. The normal equations give the same mathematical answer as QR. Why do numerical analysts insist on QR anyway?
3. The residual of a least-squares solution is always perpendicular to the image of $A$. State this geometrically: what does it mean for the residual vector relative to the columns of $A$?

## Challenge

Fit a degree-10 polynomial to 12 evenly spaced data points sampled from $\sin(x)$ on $[0, \pi]$ using (a) the normal equations and (b) Householder QR. Compute the condition number of $A^T A$ and $R$ separately. Plot both fitted curves and report the residual norm and the condition number for each method. Now repeat with 12 Chebyshev nodes instead of evenly spaced points. What changes, and why does the choice of node locations matter for the condition number?
