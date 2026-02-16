
## Header 1
Linear systems are special: It's something we can solve really well, and it covers an enourmous amount of problems. Even when you've got a non-linear system, you can isolate the linearity and solve that, and treat the non-linearity in another way.

## Header 2
#### Going from abstract linear systems to matrices

A **linear function** is anything that behaves in the way we call linear. They exist in vector spaces: A mapping $F: X \rightarrow Y$ (abstract, eg: wave functions) is linear if $F(ax + bx') = aF(x) + bF(x')$. 

Linear functions and matrices are the same thing!
* Let's say you have a basis for $X$, eg: $\{e_{1}, e_2, \dots e_n\}$, then any vector $x \in X$ can be writte uniquely as $x = a_1e_1 + \dots + a_n e_n$
* Let $\{f_{1}, f_2, \dots f_n\}$ form a basis for $Y$, any vector $y \in Y$ can be writtten as $y = b_1f_1 + \dots + b_m f_m$
* $$F(e_j)=b_{1j}f_1 + \dots + b_{mj}f_m$$
	We can write it because of linearity, but you'll notice that it's just matrix multiplication.
* $$ F(x) = F(a_1e_1 + \dots + a_ne_n) = a_1F(e_1) + \dots + a_nF(e_n) $$
	$$ = \sum_{j=1}^n a_j \sum_{i=i}^m b_{ij} f_i $$ 
	$$ = \sum_{i=i}^m f_i \left(\sum_{j=1}^n a_j b_{ij} \right) $$ 
	The thing on the right is nothing more than a matrix product. Think of the total thing as:  Representation of $F$ in $e,f$ basis $*$ (Coorrdinates of $x$ in $e$ basis) = Coordinates of $y$ in $f$ basis

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

## Can we solve
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
* $\underline{F}$ is invertible (Deterministiclaly find unique solution)
* **$F$ is singular**:
* $Im(F) \not\subseteq Y$
* $Span(\underline{F} \not\subseteq \mathbb{C}^n)$
* $Rank(\underline{F}) < n$
* $det(\underline{F}) = 0$
* There is a non-trivial subspace $Ker(\underline{F})$ such that $\underline{F}(\underline{x})=0$ for $\underline{x}\in Ker(\underline{F})$

Singular $F$ splits into two sub-cases:
1. $y \not\in Im(F) \implies$ there are no solutions
2. $y \in Im(F) \implies$ infinitely many solutions (becaues we can add something from the kernel and get another solution).

That was the mathematical part: Now we're going to look at a case where we have exact solutions: when are the solutions stable, and when do small perturbations cause it to blow up?

## Sensitivity of a Linear System of Equations

The more orthogonal the matrix is, the lower the condition number. It's ~1 for orthogonal, but as they get closer to one another, i.e, they get closer to being linearly dependent on one another, condition number increases. The webpage has a calculation of the exact condition number.

## How to build the algorithms from scratch

Construct algoithms that transforms $b=Ax$ into $x$ using a modest number of operations that are linear, invertible, and simple to compute.

**Fundamental Property:** if $M$ is invertible, then $MAx = Mb$ has the same solutions as $Ax=b$

What we know:
1. Our algorithm must have the same effect as multiplying by the inverse (you don't want to actually calculate the inverse and multiply because it introduces a lot of computational error): $A \rightarrowtail I; I \rightarrowtail A; b \rightarrowtail x$
2. Each step must be linenar, invertible
3. We first want $A = LU$ (lower and upper triangular matrices)
4. Every step, we want to take an $n\times n$ matrix, and reduce the leftmost column to be zero below the first element. Then, we just recursively continue for an $(n-1)\times(n-1)$ matrix
5. Row scaling (it's linear and invertible)
6. Row addition and subtraction is also linear and invertible

We can use 5. and 6. together: just scale the rows (if the leading term is nonzero) and subtract

"Sorry Anton, I ain't taking notes on gaussian elimination"

## lu_factorize
def lu_factorize(M):
    """
    Factorize M such that M = LU

    Parameters
    ----------
    M : square 2D array
        the 1st param name `first`

    Returns
    -------
    L : A lower triangular matrix
    U : An upper triangular matrix
    """
    m, U = M.shape[0], M.copy()
    L = np.eye(M.shape[0]) # make identity 
    for j in range(m-1):
        for i in range(j+1,m):
            scalar    = U[i, j] / U[j,j]
            U[i]     -=  scalar*U[j]
            L[i, j]   = scalar
            
    return L, U

## forward_substitute
def forward_substitute(L,b):
    '''
    Takes a square lower triangular matrix L 
    and a vector b as input, and returns the 
    solution vector y to Ly=b.
    '''
    y = np.zeros(np.shape(b))
    for i in range(len(b)):
        y[i] = ( b[i] - L[i] @ y) / L [i,i]
    return y
## backward_substitute
def backward_substitute(U,y):
    '''which takes a square upper triangular 
    matrix U and a vector y as input, and returns
    the solution vector  x  to  Ux=y.
    '''
    x = np.zeros(np.shape(y))

    for i in range(1,1+len(x)):
        x[-i] = ( y[-i] - U[-i] @ x )/U[-i,-i]

    return x

## solve_lin_eq
def solve_lin_eq(M,z):
    L,U = lu_factorize(M)
    y = forward_substitute(L,z)
    return backward_substitute(U,y)

## Header 3
Consider the matrix equation $Ax=b$
For i in I, $m: a_{11}x_1 + a_{12}x_2 + \dots + a_{in}x_n \equiv A_i^T x=b_i$. A hypersphere

If there are 2 unknowns, i.e, n=2,
$$a_{11}x_1 + a_{12} x_2 = b1$$
$$a_{21}x_1 + a_{22} x_2 = b2$$
In two dimensions, these define lines, we can find the slope and y-intercepts. The point of intersection is the solution to the system.

Assume that there's an error or uncertainity in the data, so that $||\Delta b||_\infty < \delta$. You'll have a region of uncertainity around the lines, transforming them into rectangles. Now, your solution can be anywhere in the shape that's made from the intersection of the rectangles. If we had an error in the matrix, we should have a skew (and a shift in slopes) and not a vertical shift like we do if the error is in $b$.

#### Error bounds:

1. RHS: Assume $A \hat{x} = \hat{b} \implies A(x+\Delta x) = (b+\Delta b)$. That implies:
	* $||\hat{b}|| = ||A\hat{x}|| \leq ||A||\; ||\hat{x} ||$ and so $||\hat{x}|| \geq \frac{||\hat{b}||}{||A||}$
	* $|| \Delta x || = || A^{-1} \Delta b || \leq || A^{-1}|| \; ||\Delta b ||$
	* together, we get $\frac{||\Delta x||}{||\hat{x}||} \geq \frac{||\Delta b||}{||\hat{b}||} \; ||A^{-1}|| \;|| A || = COND(A) \frac{||\Delta b||}{||\hat{b}||}$
2. Nothing here depends on matrix $A$ being exact, and we can replace $A$ here with $\hat{A}$ everywhere so we can get a calculation even if it's not exact

#### Overdetermined Systems: m > n:

Large number of equations, small number of unknowns. In general, there are _no exact solutions_.

All the places you can reach are driven by $Im(A)$, the image. There's no $x$ that you can feed in that results in you ending up outside the image.

#### How would a least squares approximate solution have to look?

**Write** $\mathbb{C}^m = Im(A) \oplus Im(A)^\perp$
Any vector space with a subspace can be broken down into the subspace and a space that's orthogonal to the subspace
**Means** $b \in \mathbb{C}^m$ can be uniquely written $b = \hat{b} + b^\perp$, with $\hat{b} \in Im(A), b^\perp \in Im(A)^\perp$
**Where** \
$$Im(A)^\perp = \{x\in \mathbb{C}^m | x^T x' = 0 \;\forall\; x' \in Im(A)\}$$ 

$$=\{x\in \mathbb{C}^m | A^T_ix=0 \text{ for } 1 \leq i \leq m \}$$

This helps us because of pythagoras:

$$||r||^2 = ||b-Ax||^2$$
$$||\tilde{b} + b^\perp - Ax||^2 = ||(\tilde{b}-Ax)+b^\perp$$
with Pythagoras (only valid in L2 norm),
$$ ||\tilde{b}=Ax||^2 + || b^\perp ||^2$$
The first is in the image, the second is in the perpendicular subspace. We can somehow set the first term to be zero, and the second term is somehow a constant.

**So:** residueal of least square is perpendicular to Im(A)