# Linear Least Squares

> *Remember the [condition number](./bounding-errors)? Here's where it bites us — the normal equations square it, and Householder reflections save us.*

## Big Ideas

* When there are more equations than unknowns, the residual inevitably has a component perpendicular to the image of $A$ — that part is irreducible, no matter how clever your solver.
* Forming the normal equations $A^T A x = A^T b$ squares the condition number, which is like photocopying a blurry photo — every generation loses quality.
* Householder reflections are orthogonal transformations (pure rotations and reflections), so they never stretch the problem and never amplify the condition number.
* QR factorization is the right algorithm not because it's elegant (though it is), but because it's the numerically stable way to get what the normal equations give you algebraically.

## Overdetermined Systems: m > n

Lots of equations, few unknowns. In general, there are _no exact solutions_.

You can only reach vectors in $Im(A)$. If $b$ isn't in the image, you're stuck — but you can get as close as possible.

### How does a least squares solution look?

Watch this trick — Pythagoras makes the whole thing click:

**Write** $\mathbb{C}^m = Im(A) \oplus Im(A)^\perp$

**Means** $b = \tilde{b} + b^\perp$, with $\tilde{b} \in Im(A)$ and $b^\perp \perp Im(A)$

$$||r||^2 = ||b-Ax||^2 = ||(\tilde{b}-Ax)+b^\perp||^2$$

Since these components are orthogonal, Pythagoras gives us:
$$ = ||\tilde{b}-Ax||^2 + || b^\perp ||^2$$

*The total error splits into a part we can control (by choosing x) and a part we can't. Minimize the first, accept the second.*

**So:** the residual of least squares is perpendicular to $Im(A)$.

[[simulation geometric-projection]]

## The Normal Equations — and Why They're Dangerous

The residual being perpendicular to the image gives us:
$$A^T(b-Ax)=0 \implies A^TAx = A^Tb$$

$A^TA$ is a nice small square matrix. But here's the catch: $\text{COND}(A^TA) = \text{COND}(A)^2$.

*QR is better because it never squares the condition number.* If $\text{COND}(A) \approx 10^8$, the normal equations square that to $10^{16}$, and **all sixteen digits of double precision disappear** — the answer is pure noise.

Why does multiplying by $Q^T$ not mess up the condition number, but multiplying by $A^T$ does? Because $Q$ is orthogonal — it's a pure rotation that doesn't stretch anything. Multiplying by $Q^T$ is like turning your head to look at the problem from a different angle. Multiplying by $A^T$ is like squishing the problem through a funnel.

## Building a Least Squares Algorithm

**Goal:** Construct the effect of $Q^T$ such that $Q^T A = R$, where $R$ has zeros below the diagonal.

### The Householder reflection: bouncing light off a mirror

Here's the beautiful part. Imagine you're holding a flashlight (that's your column vector $a$) and you want all the light pointing along the first axis $e_1$. **Place a mirror at exactly the right angle and bounce the light off it.** That's what a Householder reflection does — zeros out everything below the diagonal in one bounce.

**Building blocks:** We need _unitary_ operations that don't change vector lengths: rotations, reflections.

**The construction:**
1. **Set up the mirror.** Reflect column $a$ onto $\alpha e_1$, where $\alpha = -\text{sign}(a_1)\|a\|$ for numerical stability.
2. **Bounce the light.** $Ha = a - 2P_v a$, where $H = I - 2P_v$
3. **Find the mirror's orientation.** The projection operator is $P_v = \frac{vv^T}{v^Tv}$, and $v \propto a - \alpha e_1$.

Why not use Givens rotations instead? You can! But Householder zeros out an entire column in one shot (one mirror bounce), while Givens zeros out one entry at a time. Householder is the sledgehammer; Givens is the scalpel.

## Code

Watch this little magician at work:

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
	"""Apply Householder reflection in-place — like a mirror reflection, zap!"""
	S = -2 * np.dot(v, A) / np.dot(v, v)
	A += v[:, np.newaxis] * S[np.newaxis, :]
```

## least_squares

```python
def least_squares(A, b):
    """
    Solve min ||Ax - b||_2 using QR factorization.
    Project b onto the column space of A without squaring the condition number.
    """
    Q, R = householder_QR_slow(A)
    v = Q.T @ b                        # rotate b into Q's frame
    size = np.shape(A)[1]
    R_cropped = R[:size, :size]         # keep the upper triangular part
    v_cropped = v[:size]                # keep the part in the column space
    x = backward_substitute(R_cropped, v_cropped)  # unwrap the present

    # The residual: entries of Q^T b below the first 'size' rows
    # are the component of b orthogonal to Im(A) — the part we can't fix
    residual = np.linalg.norm(v[size:])
    return x, residual
```

> **Challenge.** Generate 20 noisy data points from $y = 2x + 3 + \text{noise}$. Fit a line using your `least_squares` function and compare with `np.polyfit`. Plot both. They should agree — but now you understand what's happening under the hood.

---

## What Comes Next

Least squares is the last linear paradise. From here on, the world is nonlinear: equations where you can't express the solution as a matrix times a vector, where the number of solutions is unknown, and where convergence is a gift rather than a guarantee.

## Check Your Understanding

1. You have 100 measurements and want to fit a degree-3 polynomial. How many columns does $A$ have, how many rows, and why is the system overdetermined?
2. The normal equations give the same mathematical answer as QR. So why do numerical analysts insist on QR?
3. The residual of a least-squares solution is perpendicular to the image of $A$. What does this mean geometrically?

## Challenge

Fit a degree-10 polynomial to 12 evenly spaced data points from $\sin(x)$ on $[0, \pi]$ using normal equations vs Householder QR. Watch the condition number explode in one case and stay tame in the other. Then repeat with Chebyshev nodes and see what changes.
