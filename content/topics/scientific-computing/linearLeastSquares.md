# Linear Least Squares



## Header 1
With given data and a desired function, determine the paramters of the function to minimize the distance to data points.


## Least Square Solution
* **Always Exists:** solution to $Ax=\tilde{b}$ is the least square solution. 
$\tilde{b}$ is the projection on the image space, and $b^\perp$ 
the projection on the orthogonal component
* **Unique if Rank(A) = n:** If the rank is less than n, there are 
infinitely many solutions: you can find one and add the kernel space 
to it to get them all
* **Normal equations**: 
$A^T_ir=0 \Rightarrow A^Tr=0 \implies A^T(b-Ax)=A^Tb-A^TAx=0$. 
The fact that the residual is orthogonal to the image means that 
for all the rows of $A$, the dot product with the residual has to be 0.
* **$A^TA$ is a small square matrix**
* The only problem is that this has a large condition number: COND$(A^TA)$ = COND$(A)^2$


We're almost at our goal, but not there yet. We've found an efficient solution (the normal equations) which are great for mathematical calculations, but we can't use them because a small error will ruin us because of the condition number.

### How do we save our significant digits (hopefully without too much work?)

Note that $Im(A)^\perp = Ker(A^T) \implies \mathbb{C}^m=Im(A)\oplus Ker(A^T)$.

### Recap
We decomposed the target space $\mathbb{C}^m$. [todo: google "Chiyo Prison Scool"] 

COND($A^TA$) = COND($A^2$), but COND($Q^TA$) = COND($A$).

We just need to construct the effect of multiplying our matrix by $Q^T$. In general, you never want to calculate $Q$ cause it can be really large (million x million).

## Building a least squares algorithm from scratch

Note: Norm $\left(\,||x||\,\right)$ always refers to the Euclidean Norm $\left(\,||x||_2\,\right)$ when we're talking about least squares.

**Goal:** Construct the effect of $Q^T$ such that $Q^T A = B$, where $B$ is a matrix with the bottom part being 0 and the upper part being upper triangular.

**Building blocks:** We want to perform _unitary_ operations: i.e, do things that don't change the length of vectors: operations like rotation, reflection (Not translations, because although it doesn't change lengths it changes the norm).
* 2D: Rotations and reflections
* 3D: Rotations, reflections and inversions (like reflection through a point)
* Higher dimensions: lots more (symmetries of n-spheres), permutations (everywhere)

**What do we want to build?** Similar to LU, we want to take a column and eliminate everything below the diagonal. Construct a reflection operation $H$ (for Householder), which is a reflection. The norm of the entire column must be the same though, and so the top element (the diagonal) must contain $||a||$ concentrated in it.

First, we reflect vector (column) $a$ onto basis vector $e_1$. Consider a mirror, which is the angle bisector in between $a$ and $e_1$. We could reflect it onto $e_1$ or even onto the negative side, $-e_1$. Call the two mirrors $v^+$ and $v^-$

The operation needs to transform $\vec{a} \rightarrowtail \alpha \vec{e_1}$, where $\alpha= \pm ||a||$

$$Ha = a - 2 Pva$$
$$ H = I - 2Pv \qquad v=v^+ \text{ or } v^-$$

How to find $v^+, v^-$?

Projection operator, $P_v a = \frac{v^T a}{||v||^2} v= \frac{v^T a}{v^T v} v = \left(\frac{vv^T}{v^tv} a\right)$
$$\implies P_v = \frac{vv^T}{v^Tv}$$

$$\alpha e_1 = Ha = a-2\frac{v^Ta}{v^Tv}v$$
$$\underbrace{\frac{2v^Ta}{v^Tv}} v =a-\alpha e_1$$
First part is just a scalar number, the rest are vectors. It gets normalized away if we choose $||v||=1$

Thus, $v \propto a - \alpha e_1$. Just subtract $\alpha$ from the first entry in the column

Now we can plug that into the equation for $H$, and we're done.

## Code
def HouseholderQR(A, b):
	m, n = A.shape
	R = A.astype(float)
	b_tilde = b.astype(float)
	for k in range(n):
		a = R[k:m, k]
		if (np.dot(a[1:],a[1:]) == 0):
			continue
		v = reflection_vector(a)
		reflect_columns(x, R[k:m, k:n])
		reflect_columns(v, b_tilde[k:m])
	return R, b_tilde

def reflect_columns(v, A):
	S = -2 * np.dot(v, A)
	A -= v[:, NA] * S[NA, :]

## Header 3
the last line RHS is the same as `np.outer(S,V)`

This then yields a much better least squares optimizer:


## least_squares
def least_squares(A, b):
    '''
    Solves Ax = b, for x
    '''
    Q, R = householder_QR_slow(A)
    v = Q.T @ b
    size = np.shape(A)[1]
    R_cropped = R[:size,:size]
    v_cropped = v[:size]
    x = backward_substitute(R_cropped, v_cropped)
    
    residual = max(abs(v[size-1:]))
    return x, residual
