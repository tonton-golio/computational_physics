import streamlit as st
st.set_page_config(page_title="Scientific Computing", 
	page_icon="🧊", 
	layout="wide", 
	initial_sidebar_state="collapsed", 
	menu_items=None)
st.title("Scientific Computing")

with st.expander('Bounding Errors', expanded=False):
	st.markdown(r'''# Approximations & Errors
Take in inputs (measurements/ prior computation/ physical constants), do something with a computer (finite no. representations), and return an output. 

Computer representations include _Floating point numbers_: $(-1)^{\text{sign}} * 2^{\text{exponent}} * \text{mantissa}$

**Notation**: the input is our computer representation is \hat{x}, the true input is x (note: x is a vector). \Delta x is the difference. The problem we solve has an _ideal solution_, f(x); we use an algorithm which is an approximation to this called \hat{f}. We end up with \hat{f}(\hat{x}) as the _actual solution_. We want to analyze the relation between the actual and ideal solutions.
$$
\text{(total forward error): } E_{tot}  = \hat{f}(\hat{x}) - f(x)\\
= \hat{f}(\hat{x}) + (- f(\hat{x}) + f(\hat{x})) - f(x)\\
= E_{Comp} (Computational Error) - E_{Data} (Propogated Data Error)
$$

Computational error can be split int truncation error $E_{trunc}$ and rounding error E_{round}.

**Truncation error** contains:
* Simplifications of the physciial model (frictionless, etc)
* Finite basis sets
* Truncations of infinite series

etc: call the things you do to take your abstract problem to something you can actually solve.

**Rounding error** contains everything that comes from working on a finite computer
* Accumulated rounding error (from finite arithmetic)

## Example

Computational error of first order finite difference: $$ f'(x) = \frac{f(x+h) - f(x)}{h} \def \hat{f'}(x) $$
Taylor expand:
$$ f(x+h) = f(x) + h*f'(x) + \frac{h^2}{2} f''(\theta), \qquad \lvert \theta - x \rvert \leq h $$
$$ \frac{f(x+h) - f(x)}{h} = f'(x) + \frac{h}2 f''(\theta) $$
$$ \hat{f'}(x) - f'(x) = \frac{h}2 f''(\theta) $$
$$ M \def \sup{\theta - x = h} \lvert f''(\theta) \rvert $$
$$ E_{trunc} = \hat{f'}(x) - f'(x) \leq \frac{M}2 h \quad \sim O(h) $$

What about rounding error? Assume that for $f$, it's bounded by $\epsilon$
$$ E_{round} \leq \frac{2\epsilon}h \quad \sim O\left(\frac1h\right) $$ 
(comes from floating point somehow... use significant digits?)

If you decrease $h$, you decrease truncation error but increase rounding error

$$ E_{comp} = \frac{M}2 h + \frac{2\epsilon}h $$
What value of $h$ minimizes it? Differentiate

$$ 0 = \frac{M}2 - \frac{2\epsilon}{h^2} $$
$$ h^2 = \frac{4\epsilon}M $$
$h$ can't be negative, so
$$ h = \frac{2 \sqrt{\epsilon}}{\sqrt{M}} $$
(Note that $\epsilon$ is a bound)

**Propagated Data Error**: The problem can either expand or contract the error from your data, and it's importat to understand what it does

Absolute forward data error = $f(\hat{x}) - f(x) \equiv \Delta y$

Relative forward data error = $ \frac{\Delta y}{y} = \frac{f(\hat{x}) - f(x)}{f(x)}$

We also use a _Condition Number_: How much a change in the data affects a change in the result.
$$ COND(f) = \frac{\lvert \Delta y/y \rvert}{ \lvert \Delta x/x \rvert} = \frac{\lvert x \Delta y \rvert}{ \lvert y \Delta x \rvert} $$

It may be intutive that if you start with 4 digits of input, your ouput will be correct to 4 digits at max. This isn't the case: Consider a function $f(x) = x^{\frac{1}{10}}$, and let's analyze the errors.

$$ E_{data} = f(\hat{x}) - f(x) $$
$$ (x+\Delta x)^\frac{1}{10} - x^\frac{1}{10} $$
The relative error will be
$$ E^{rel}_{data} = \frac{(x+\Delta x)^\frac{1}{10} - x^\frac{1}{10}}{x^\frac{1}{10}} $$
Now Taylor expand
$$ = \frac{x^\frac1{10} + \Delta x x^\frac{-9}{10} - x^\frac{1}{10}}{x^\frac{1}{10}} + O(\frac{\Delta x^2}{x^\frac1{10}})$$
$$ \Delta y / y = \frac1{10} \Delta{x} /x + O(quadratic)$$
You can have an additional significant digit in the output: start with 3, end up with 4, etc

**In general**:
* $\sqrt{x}$ has 1 more significant bit as compared to $x$
* $x^\frac1{10^n}$ has n more decimal significant digits
* x^2 is 1 fewer bit significant
* x^10^n has n fewer decimal sig digits 

But information theory tells us that information cannot be gained out of nowhere: what's going on?
''')
	
	st.markdown('''
	### Bounding Errors: 

	https://www.youtube.com/watch?v=GFhhRdF54eI
	
	**Sources of approximation** include modelling, 
	empirical measurements, previous computations, truncation/discretization, rounding.

	**Absolute error** and **relative error** are different in the obvious manner:
	''')
	st.latex(r'''
		\begin{align*}
		\text{abs. error } &= \text{approx. value }-\text{true value}\\
		\text{rel. error } &= \frac{\text{abs. error}}{\text{true value}}.
		\end{align*}
	''')

	st.markdown('''
	**Data error and computational errror**, the hats indicate an approximation;
	''')

	st.latex(r'''
	\begin{align*}
	\text{total error} &= \hat{f}(\hat{x})-f(x)\\
	&= \left(\hat{f}(\hat{x})-f(\hat{x})\right) &+&\left(f(\hat{x})-f(x)\right)\\
	&= \text{computational error} &+& \text{propagated data error}\\
	E_\text{tot} &= E_\text{comp} &+& E_\text{data}
	\end{align*}''')

	st.markdown(r'''
	**Truncaiton error and rounding error** are the two parts of computational error. 
	Truncation error stems from truncating infinite series, or replacing derivatives 
	with finite differences. Rounding error is like the error from like floating point accuracy.

	Truncation error, $E_\text{trunc}$ can stem from; 
	* simplification of physical model
	* finite basis sets
	* truncations of infinite series
	* ...

	*Example:* computational error of $1^\text{st}$ order finite difference
	''')
	st.latex(r'''
	\begin{align*}
	f'(x) \approx \frac{f(x+h)-f(x)}{h}\equiv\hat{f'(x)}\\
	f(x+h) = f(x)+hf'(x)+ \frac{h^2}{2}f''(\theta), |\theta-x|\leq h\\
	\frac{f(x+h)-f(x)}{h} = f'(x) + \frac{h}{2}f''(\theta)\\
	\hat{f'(x)} - f'(x) = \frac{h}{2}f''(\theta), \text{let} \equiv \text{Sup}_{|\theta-x|\leq h} (f''(\theta))\\
	E_\text{trunc} = \hat{f'(x)}-f'(x)\leq \frac{M}{2}h\sim O(h)
	\end{align*}''')

	st.markdown(r'''But what about the rounding error? 
		(assume R.E. for $f$ is $\epsilon \Rightarrow E_\text{ronud} \leq \frac{2\epsilon}{h}\sim 0(\frac{1}{h})$
		''')

	st.latex(r'''
	\begin{align*}
		E_\text{comp} = \frac{M}{2}h + \frac{2\epsilon}{h}\\
		0 = \frac{d}{dh}E_\text{comp} = \frac{M}{2}-\frac{2\epsilon}{h^2}\\
		\frac{M}{2} = \frac{2\epsilon}{h^2} 
		\Leftrightarrow h^2 = \frac{4\epsilon}{M}\Leftrightarrow h_\text{optimal} = 2\sqrt{\frac{\epsilon}{M}}
	\end{align*}''')

	try:
		st.image('https://farside.ph.utexas.edu/teaching/329/lectures/img320.png')
	except:
		st.image('assets/images/errors.png')


	st.markdown(r'''
	**Forward vs. backward error**

	foward error is the error in the output, backward error is the error in the input.

	**Sensitivity and conditioning**

	Condition number: $COND(f) \equiv \frac{|\frac{\Delta y}{y}|}{|\frac{\Delta x}{x}|} = \frac{|x\Delta y|}{|y\Delta x|}$

	**Stability and accuracy**

	**Floating point numbers**, a video from like 8 years ago by numberphile: 
	https://www.youtube.com/watch?v=PZRI1IfStY0.
	*floating point is scientic notation in base 2*. 

	Another video: https://www.youtube.com/watch?v=f4ekifyijIg. 
	* Fixed points have each bit correspond to a specific scale.
	* floating point (32 bit) has: 1 sign bit (0=postive, 1=negative), 8 exponent bits, 
	and 23 mantissa bits. 

	* another video on fp addition: https://www.youtube.com/watch?v=782QWNOD_Z0

	overflow and underflow; refers to the largest and smallest numbers that can be 
	contained in a floating point.

	**Complex arithmatic**

	''')

with st.expander('Linear Equations', expanded=False):
	st.markdown(r"""
Linear systems are special: It's something we can solve really well, and it covers an enourmous amount of problems. Even when you've got a non-linear system, you can isolate the linearity and solve that, and treat the non-linearity in another way.

## Going from abstract linear systems to matrices

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

## How to solve linear systems of equations

Find all $x \in X$ such that $F(x) = y$ (some known $y\in Y$)
1. Pick basis sets $\{e_1, \dots, e_n\}$ for $X$, $\{f_1, \dots, f_m\}$ for $Y$
2. Write down matrix $F$ in $e,f$ basis and RHS $y$ in $f$ basis
3. Solve matrix equation $Fx=y$, where $F$ and $y$ are known
4. Dot result $x$ with $e$ basis to get the result in the problem domain $x=x_1e_1+\dots+x_ne_n$

## Can we solve $F(x)=y$?
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

## Sensitivity of a Linear System of Equations:

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

Sorry Anton, I ain't taking notes on gaussian elimination""")
	st.markdown(r"""Consider the matrix equation $Ax=b$
For i in I, $m: a_{11}x_1 + a_{12}x_2 + \dots + a_{in}x_n \equiv A_i^T x=b_i$. A hypersphere

If there are 2 unknowns, i.e, n=2,
$$a_{11}x_1 + a_{12} x_2 = b1$$
$$a_{21}x_1 + a_{22} x_2 = b2$$
In two dimensions, these define lines, we can find the slope and y-intercepts. The point of intersection is the solution to the system.

Assume that there's an error or uncertainity in the data, so that $||\Delta b||_\infty < \delta$. You'll have a region of uncertainity around the lines, transforming them into rectangles. Now, your solution can be anywhere in the shape that's made from the intersection of the rectangles. If we had an error in the matrix, we should have a skew (and a shift in slopes) and not a vertical shift like we do if the error is in $b$.

## Error bounds:

1. RHS: Assume $A \hat{x} = \hat{b} \implies A(x+\Delta x) = (b+\Delta b)$. That implies:
    * $||\hat{b}|| = ||A\hat{x}|| \leq ||A||\; ||\hat{x} ||$ and so $||\hat{x}|| \geq \frac{||\hat{b}||}{||A||}$
    * $|| \Delta x || = || A^{-1} \Delta b || \leq || A^{-1}|| \; ||\Delta b ||$
    * together, we get $\frac{||\Delta x||}{||\hat{x}||} \geq \frac{||\Delta b||}{||\hat{b}||} \; ||A^{-1}|| \;|| A || = COND(A) \frac{||\Delta b||}{||\hat{b}||}$
2. Nothing here depends on matrix $A$ being exact, and we can replace $A$ here with $\hat{A}$ everywhere so we can get a calculation even if it's not exact

## Overdetermined Systems: m > n:

Large number of equations, small number of unknowns. In general, there are _no exact solutions_.

All the places you can reach are driven by $Im(A)$, the image. There's no $x$ that you can feed in that results in you ending up outside the image.

## How would a least squares approximate solution have to look?

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

## Least Square Solution:
* **Always Exists:** solution to $Ax=\tilde{b}$ is the least square solution. $\tilde{b}$ is the projection on the image space, and $b^\perp$ the projection on the orthogonal component
* **Unique if Rank(A) = n:** If the rank is less than n, there are infinitely many solutions: you can find one and add the kernel space to it to get them all
* **Normal equations**: $A^T_ir=0 \implies A^Tr=0 \implies A^T(b-Ax)=A^Tb-A^TAx=0$. The fact that the residual is orthogonal to the image means that for all the rows of $A$, the dot product with the residual has to be 0.
* **$A^TA$ is a small square matrix**
* The only problem is that this has a large condition number: COND$(A^TA)$ = COND$(A)^2$

We're almost at our goal, but not there yet. We've found an efficient solution (the normal equations) which are great for mathematical calculations, but we can't use them because a small error will ruin us because of the condition number.

## How do we save our significant digits (hopefully without too much work?)

Note that $Im(A)^\perp = Ker(A^T) \implies \mathbb{C}^m=Im(A)\oplus Ker(A^T)$.

Thus, we can write (see james' notes)""")
	
with st.expander('Linear Least Squares', expanded=False):
	st.markdown(r"""# Least Squares

## Recap

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

```python
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

the last line RHS is the same as `np.outer(S,V)`""")
with st.expander('Eigensystems', expanded=False):
	st.markdown(r"""
# Eigensystems

## 

Any stable physical/chemical system can be described with an eigensystem. Wavefunctions in quantum mechanics are a typical example.
$$Ax = \lambda x$$
$A$ is the operator (mapping $\mathbb{C}^n \rightarrow \mathbb{C}^n$), $x$ an eigevector, $\lambda$ an eigenvalue.

For any fixed $\lambda$, you've got a linear system
$$(A-\lambda I)x = 0$$
which has non=trivial ($x\neq0$) solutions if 
$$det(A-\lambda I) = 0$$
Eigenvectors always come in subspaces; $E_\lambda = \{x\in \mathbb{C}^n\vert Ax=\lambda x\}$; and the spaces they span are called _eigenspaces_. The spaces are the kernels of the above matrix. If you have two eigenvectors, then their sum is an eigenvector as well.

If we can find $\lambda_1, \lambda_2, \dots \lambda_p$, such that $\mathbb{C}^n=E_{\lambda_1}\oplus E_{\lambda_2}\oplus \dots \oplus E_{\lambda_p}$, then we fully understand the actions of $A$

Notice that $det(A-\lambda I)=0$ is a polynomial in $\lambda$, called the _characteristic polynomial_. By the **fundamental theorem of algebra**, any $n$ degree polynomial has exactly $n$ complex root, with multiplicity, such that you can write the polynomial as $P(\lambda)=c_0(\lambda-\lambda_1)(\lambda-\lambda_2)\dots(\lambda-\lambda_n)$ where $\lambda_1,\lambda_2,\dots,\lambda_n\in\mathbb{C}$

The issue is that we have two types of multiplicity for an eigenvalue $\lambda$
1. **Algebraic multiplicity:** How many times an eigenvalue, $\lambda_i$, appears in the characteristic polynomial.
2. **Geometric multiplicity:** You can have a multiple root that has only a 1D eigenspace. The dimension of your eigenspace is the number of linearly independent eigenvectors.

Geometric multiplicity $\leq$ algrebraic. Two cases:
* They're equal: _Non-defective_: $\sum_{\lambda\in Sp(A)} Dim(E_\lambda) = n$. This is the good case.
* _Defective_: $\sum_{\lambda\in Sp(A)} Dim(E_\lambda) \lt n$. Nothing works here, and so we'll conveniently ignore them

The $Sp(A)$ is the _spectrum_ of operator $A$. It's the set of all the eigenvalues of $A$. $Sp(A)=\{\lambda\in\mathbb{C}\vert P(\lambda=0)\}$

Thankfully, most cases in natural sciences are not defective, and so we can analyze them. Non-defectiveness is guaranteed if:
* $A$ is _normal_: $A^H A = AA^H$ ($A^H$ is the hermitian adjoint of $A$)
* $\lambda_1,\lambda_2,\dots,\lambda_n$ are distinct
* The best case: $A$ is Hermitian: $A=A^H$. In this case, we are guaranteed to get an orthonormal basis of eigenvectors.

Then $A = U \Lambda U^H$ ($\Lambda = Sp(A)$)

## The power method

Let $A:\mathbb{C}^n \rightarrow \mathbb{C}^n$ be a non-defective linear operator. $A=T\Lambda T^{-1}$, and let's order the eigenvalues in descending order: $\vert\lambda_1\vert\geq\vert\lambda_2\vert\geq\dots\geq\vert\lambda_n\vert$ ($\geq$ because the eigenvectors are not necessarily distinct)

Because eigenvectors $\{t_1,t_2,\dots,t_n\}$ forms a basis for $\mathbb{C}^n$, for any $x\in\mathbb{C}^n$,
$$Ax = A(\tilde{x}_1t_1+\tilde{x}_2t_2,\dots,\tilde{x}_nt_n)$$
By the linearity of $A$, we get
$$= \tilde{x}_1At_1+\tilde{x}_2At_2,\dots,\tilde{x}_nAt_n$$
We know what $A$ does to eigenvectors:
$$= \tilde{x}_1\,\lambda_1\,t_1+\tilde{x}_2\,\lambda_2\,t_2,\dots,\tilde{x}_n\,\lambda_n\,t_n$$
If we apply $A$ $k$ times, then we'd get
$$A^kx=\tilde{x}_1\,\lambda_1^k\,t_1+\tilde{x}_2\,\lambda_2^k\,t_2,\dots,\tilde{x}_n\,\lambda_n^k\,t_n$$
$$=\lambda_1^k\left(\tilde{x}_1\,t_1+\tilde{x}_2\,\left(\frac{\lambda_2}{\lambda_1}\right)^kt_2,\dots,\tilde{x}_n\,\left(\frac{\lambda_n}{\lambda_1}\right)^kt_n\right)$$
But since $\lambda_1$ is the biggest, the ratio of the other brackets go to 0 as $k\to\infty$
$$= \lambda_1^k\left(\tilde{x}_1t_1 + O\left(\lvert\frac{\lambda_2}{\lambda_1}\rvert\right)\right)$$
as $\lambda_2$ is the second largest eigenvalue
$$\lim_{k\to\infty}\frac{A^kx}{\Vert A^k x \Vert} = t_1, \qquad\text{if }\tilde{x}_1\neq0 \text{ and } \lambda_2<\lambda_1$$
If $\lambda_1=\lambda_2=\dots=\lambda_n$, it still gives the eigenvector to $\lambda_1$: Namely, the projection $t_{\lambda_1}=P_{\lambda_t}x$

### Power method algorithm

```python
def PowerIterate(A, x_0):
    x = x_0
    while (not_converged(A, x)):
        x = A @ x
        x /= np.linalg.norm(x)
    return x
```

You can check for convergence by seeing if it's an eigenvector. Either check if $Ax$ is parallel to $x$, or use the Rayleigh equation (which we'll cover in lecture 6).

## Something else he's squeezed in

Let $A, B$ be non-defective with the same eigenvectors $T$.

$A=T\Lambda_AT^{-1}$, $B=T\Lambda_BT^{-1}$. Then
1. $A+B=T\Lambda_AT^{-1}+T\Lambda_BT^{-1}=T\left(\Lambda_A+\Lambda_B\right)T^{-1}$
2. $AB=T\Lambda_AT^{-1}T\Lambda_BT^{-1}=T\Lambda_A\Lambda_BT^{-1}$
3. $A+\sigma I=T\Lambda_AT^{-1}+T\sigma IT^{-1}=T\left(\Lambda_A+\sigma I\right)T^{-1}$

Because we have these, we can construct polynomials.

$$P(A)=c_0J+c_1A+c_2A^2+\dots+c_dA^d$$
$$=\sum_{k=0}^d c_kA^k=\sum_{k=0}^d T \Lambda_A T^{-1}$$
Using the linearity of $T$:
$$=T \left(\sum_{k=0}^d\Lambda_A\right) T^{-1} = T\; P(A)\; T^{-1}$$

Now, we use _math_

Let $f:\mathbb{C}\rightarrow\mathbb{C}$ be continuous, then there exists a sequence of $d$ degree polynomials $P_d(\lambda)$ such that $f(\lambda)=\lim_{d\to\infty} P_d(\lambda)$
$$f(A) = \lim_{d\to\infty} P_d(\Lambda_A) = \lim_{d\to\infty}\left(T\;P(\Lambda_A)\;T^{-1}\right)$$
$$ =T\left(\lim_{d\to\infty}P(\Lambda_A)\right)T^{-1} $$
* Gershgorin centers
* rayleigh quotient
* power iterate (gives us the greatest eigenvalue)
* rayleigh iterate""")
with st.expander('Nonlinear Equations	Optimization', expanded=False):
	st.markdown(r""""# Nonlinear equations

[Get notes from fabri, I can't take stuff down properly]

## Notation:

* $f(x) = 0$: 1 dimensional stuff. $f:\mathbb{R} \to \mathbb{R}$
* $F(x) = 0$: Higher dimensional stuff, $F$ is a matrix an $x$ a vector. $F:\mathbb{R}^n\to\mathbb{R}^m$

Complex numbers are complicated, so we'll only work with reals.


## Introduction

Emergent macroscopic behaviour comes out of high dimensionality of linear systems. For example, you don't figure out the aerodynamics of a plane by using the schrodinger equation for every atom. You can make a simpler theory based on the emergent behaviour, which could be non-linear despite the underlying rules being linear. You could also just make up a non-linear problem, like in economics.

**Linear Systems:** 😊
* We know exactly how many solutions exist (by looking at the matrix's rank)
* We have methods to find exact solutions (if they exist) or approximate solutions (if they don't exist).
* We can find the full solution space of the problem by adding the kernel
* We can routinely solve for billions of dimensions: it's very efficient

**Non-linear systems:** 💀
* No idea how many (if any) solutions. All we can hope for is rules of thumb, heuristics, and we can look for something that works as much as possible and fails rarely. We won't get great results in a finite number of steps, sometimes it gets closer and sometimes it doesn't.
* No fail-proof solvers
* No way of knowing if we've found all the solutions
* Even 1 dimensional solutions can take ages

## How many solutions?

Globally, *anything is possible*. $e^{-x}$ has no solutions, but $(e^{-x}-\delta)$ (where $\delta$ is a small number) does. $\sin(x)$ has countably infinite solutions, while $\text{erf}(x) -\frac12$ has uncountably infinite solutions.

Locally, we can sometimes work it out. In 1D, we can look at where $f(x)$ changes sign, and we can assume that there's a root in between (Intermediate Value Theorem, assuming the function is continuous). In particular, there are $>1$ roots in the region, and there are an odd number of roots.

## General algorithm construction scheme

In general, you want to find an algorithm which does the following:
1. Find invariant that guarantees existence of a solution in the search space
2. Design operation that preserves 1. and shrinks search space

It's a tried and true scheme for coming up with algorithms, Euclid did it thousands of years ago, and we'll do it now. Let's use this to build an equation solver on a bracket. (A *bracket* is an interval in which $f(x)$ changes sign).

## Bisection method
1. Our invariant: $a<b$ and $\text{sign}(f(a)) \neq \text{sign}(f(b))$
2. Set $m = \frac{a+b}{2}$ and evaluate $S_m = \text{sign}(f(m))$
3. If $S_m == S_a$: $a=m$
4. If $S_m == S_b$: $b=m$
5. If $S_m == 0$: We've found the root

Note: you should not use $m = \frac{a+b}{2}$ because of floating point error: use  $m = a + \frac{b-a}{2}$

```python
def bisection(f, a, b, tolerance):
    n_steps = np.log2((b - a) / tolerance)
    S_a, S_b = sign(f(a)), sign(f(b))
    for i in range(nsteps):
        m = a + (b - a) / 2
        s_m = sign(m)
        if S_m == S_a:
            a = m
        else:
            b = m
    return m
```

## Conditionings

The conditioning for evaluating $f(x)$ is approximately $\vert \frac{x f'(x)}{f(x)} \vert$ (taylor expansion) and the absolute error is $\vert f'(x)\vert$ When evaluating $f^{-1}$, the conditional number is $\approx \vert  \frac{f(x)}{x f'(x)} \vert$ aand the absolute error is $\vert\frac{1}{f'(x)} \vert$

As you get a high sensitivity in the inverse, you get a low sensitivity in the inversion and vice-versa. The function doesn't need to have an inverse in order to find the inverse in a local region.

We're trying to look for $f(x)=0$: when we're close to zero, we never use the relative accuracy but always the absolute one.

## Convergence

$e_{k}$ (the error at the $k^{\text{th}}$ step) $= x_k - x^*$

$$E^k_{rel} = \frac{e_k}{x^*}  = \frac{x_k - x^*}{x^*}$$

We need to look at the number of significant bits, because it's exact unlike significant decimal digits.

$$\text{bits/step} = -\log_2(E^{k+1}_{rel}) - \left(-\log_2(E^k_{rel}) \right) $$
$$ = \log_2\left(\frac{\frac{x_k - x^*}{x^*}}{\frac{x_{k+1} - x^*}{x^*}} \right) $$
$$ = \log_2\left(\frac{\vert x_k - x^*\vert}{\vert 
x_{k+1} - x^*\vert} \right) $$
$$ = \log_2\left(\frac{\vert e_k\vert}{\vert e_{k+1}\vert}\right) $$
$$ = -\log_2\left(\frac{\vert e_{k+1}\vert}{\vert e_k\vert}\right) $$

If $\lim_{k\to 0} \frac{\vert e_{k+1}\vert}{\vert e_k\vert^r} = c$, and $0 \leq c \lt 1$, method converges with rate r=1 $\implies$ linear, r=2 $\implies$ quadratic, etc

## Fixed point solvers

A systeamtic approach that works (also in n-dimensions). It uses the fixed point theorem: here we'll use _Banache's theorem_. It doesn't just hold in a vector space, but even in a metric space.

Let $S$ be  closed set $S \sub \mathbb{R}^n$ (todo change to proper subset)
and $g:\mathbb{R}^n \to \mathbb{R}^n$, if there exists $0\leq c \lt 1$ such that
$$\Vert g(x) - g(x') \Vert \leq c\Vert x - x'\Vert$$
for $x, x' \in S$, the we call $g$ a _contraction_ on $S$ and we are guaranteed a solution to $g(x)=x$ on $S$, which is
$$x^k = \lim_{k \to \infty} g^k(x_0)$$
for any $x_0 \in S$

Question: Can we transform "$f(x)=0$" to "$g(x)=x$"? The answer is yes, it's easy, but most choices are terrible.

For example, if you pick $g(x) = x - f(x)$, you usually repel solutions. Look at example 5-8 in the book, it gives 4 different ways of rewriting, some are repulsors and some attracters.

How can we make it attractive? Let's analyze the error (in 1D, because it's easier):

$$\vert\, e_{k+1} \,\vert = \vert\, x_{k+1} - x^* \,\vert $$
$$ = \vert\, x_{k+1} - g(x^*) \,\vert $$
$$ = \vert\, g(x_k) - g(x^*) \,\vert $$

Now we bring in the _Mean value theorem_: $\exists \theta \in [x_{k+1}, x^*]$ for which
$$ = \vert\,g'(\theta) (x_k - x^*)  \,\vert$$
$$ = \vert\,g(\theta)\,\vert \vert\, (x_k - x^*)  \,\vert$$
If we can bound $Sup_\theta \vert g'(\theta) \vert \leq c$

$$ = c  \vert (x_k - x^*)  \vert$$
$$ = c\, e_k $$

By continuity, if $\vert g'(x) = x \lt 1$, then for every $0 \lt \epsilon$, there exists $\delta$ such that $\vert g'(x) \vert \leq c + \epsilon$ when $\vert x-x^*\vert \lt \delta$

Thus, if $\vert g'(x^*) \vert \lt 1$, then $g$ is a contraction around $x^* \to x^*$

If $g'(x^*) =0$, then by taylor expanding, we can see that
$$g(x) = g(x^*) + 0 + \underbrace{g''(\theta)} (x - x^*)^2$$
by MVT again, $\exists\, \theta \in [x, x^*]$
$$ \implies \vert e_{k+1} \vert = \vert g(x_k) - g(x^*) \vert $$
$$ = \vert g''(\theta) \vert \vert x_k-x^*\vert^2 $$
$$ \lt \vert g''(x) \vert  \vert e_k \vert^2 $$

If we set $g$ in this way, we have quadratic convergence: the number of bits you gained is squared every timme. The closer you get, the quicker you converge.

## Newton's Method

Set $0 = f(x_k) + \Delta x_k f(x_k) + \mathcal{O}(\Delta x ^2)$
$$ \Delta x = \frac{-f(x_k)}{f'(x_k)} $$
$$ \implies x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)} $$
$\left(g(x) = x -\frac{-f(x_k)}{f'(x_k)}\right)$
$$g'(x) = 1 - \frac{f'(x)f'(x)}{f'(x)^2} + \frac{f(x) f''(x)}{f'(x)^2} $$
$$ = \frac{f(x) f''(x)}{f'(x)^2} $$

Which is Newton's method: you have to start close enough, but once you do it converges rapidly
""")

	st.write("""
# Nonlinear solvers


## Fixed point iteration equation solvers (Recap)

We can transform $f(x)=0$ to $g(x)=x$ in multiple ways, but we need to pick one that doesn't blow up in our face. We can converge to our solution with $x_{k+1} = g(x_k)$. Solutions are $x^* = g(x^*)$

This doesn't require anything to be one dimensional, it holds in higher dimensional space as well.

First, let's suppose $|g'(x)| \leq c \lt 1$ on a ball $k_\delta$ of radius $\delta$.

$$ \Vert e_{x_k + 1}\Vert = \Vert x_{k+1}-x^* \Vert$$
$$ = \Vert x_{k+1}-g(x^*) \Vert$$
$$ = \Vert g(x_k)-g(x^*) \Vert$$

Because of the _mean value theorem_, there exists an $x' \in [x_k, x^*]$

$$ = \Vert g'(x')(x_k - x^*) \Vert$$
$$ \leq \Vert g'(x')\Vert\,\Vert(x_k - x^*) \Vert $$

If $x_k \in k_\delta:$

$$ \leq c \,\Vert(x_k - x^*) \Vert $$
$$ \leq c^k \,\Vert(x_0 - x^*) \Vert $$

We can build the ball $k_\delta$ from continuity of $g'$:

**Continuity**: A function $f: X\to Y$ is continuous if for any $\epsilon>0 \quad \exists \quad \delta >0$ such that 
$$\Vert x - x' \Vert \lt \delta \implies \Vert f(x) - f(x') \Vert < \epsilon$$

Assume $|g'(x^*)| <1$, then
* Pick $0 < \epsilon<1-|g'(x^*)|$ and set $c = |g'(x^*)| + \epsilon$
* Then by continuity of $g'$ at $x^*$, there is a $\delta>0$ such that $\Vert x - x' \Vert \lt \delta \implies \Vert f(x) - f(x') \Vert < \epsilon$
* $\Vert g'(x) \Vert - \Vert g'(x^*)\Vert\lt\Vert g'(x) - g'(x^*)\Vert$ so $\Vert g'(x) \Vert \leq \Vert g'(x^*)\Vert + c$, where $c<1 \forall x\in k_\delta$

## Constructing quadratic convergence:

If $g'(x^*) = 0$ then the second term in the taylor expanion is 0

$$ g(x) = g(x^*)+g'(x^*)(x-x^*)+\frac{g''(x^*)(x-x^*)^2}{2} $$

By continuity, we can find a ball $k_\delta$ for which $\Vert g''(x)\Vert <\bar{c}$, where $\bar{c}=\Vert g''(x^*)\Vert + \epsilon$.

$$ \Vert e_{k+1} \Vert = \Vert g(x_k)-g(x^*)\Vert $$
$$ \leq \bar{c}^2\Vert x_k - x^* \Vert^2  $$
$$ \leq \bar{c}^k\Vert x_0 - x^* \Vert^{2^k}  $$

**Requirements for g:**

* $|g'(x^*)|<1$ at all solutiond $x^*$
* if $0<|g'(x^*)|<1$, we get linear convergence: $\vert e_{k+1} \vert\leq c^k | e_0|$
* if $0=|g'(x^*)|$, we get quadratic convergence: $\vert e_{k+1} \vert\leq \bar{c}^k | e_0|^{2^k}$


## Newton's method

A quadratically convergent fixed point iteration solver. Take a point, take the tangent of the curve at that point, your new point is the tangent's x-intercept. Intuition: find zero for linear approximation, set as next $x$.

Talor expand:
$$ f(x_{k+1}) = f(x_k) + f'(x_k) (x_{k+1}-x_k) + \mathcal{O}(\vert x_{k+1}-x_k\vert^2)$$
$$ \approx f(x_k) + f'(x_k) (x_{k+1}-x_k)$$
$$ 0 \approx f(x_k) + f'(x_k) \Delta x_{k+1}$$
$$ f'(x_k) \Delta x_{k+1} = f(x_k)$$
Remember the above line, cause we'll use it in higher dimensions as it's solving linear systems
$$ \Delta x_k = -\frac{f(x_k)}{f'(x_k)}  $$
This works in 1d, but is not general because we're dividing by a matrix (the jacobian) and not a number
$$ \implies g(x_k) = x_k + \Delta x_k = x_k - \frac{f(x_k)}{f'(x_k)}$$

if $f(x^*) = 0$,
$$g(x^*) = x^* - \frac{f(x)}{f'(x)} = x^*$$
Also,
$$ g'(x) = \frac{d}{dx}\left(x - \frac{f(x)}{f'(x)}\right)$$
$$ 1 - \frac{f'(x)f'(x) + f(x)f''(x)}{f'(x)^2} $$
$$  = - \frac{f(x)f''(x)}{f'(x)^2}$$
$$ \to 0 \quad \text{when}\quad f(x)=0$$

## Quasi - Newton/Secant method

These are methods that try to replicate the wonderful properties of Newton's method, but without having to evaluate the derivative. In higher dimensions, you don't want to be evaluating the derivative cause it's a massive matrix.

**Method**: Use secant (finite difference) instead of tangent (derivative).

$$f'(x) \approx \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$$
In 1D, we have the following equation for $\Delta x$
$$\Delta x_{k+1} = -\frac{f(x_k)}{\hat{f}'(x_k)}$$
where
$$\hat{f}'(x_k) = \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$$
For higher dimensions, we want to write it like
$$ \hat{f}'(x_k) \Delta x_{k+1} = - f(x_k)$$
Cause we'll solve a linear system instead

## Going to higher dimensions:

Notation: $f: \mathbb{R}\to\mathbb{R}$, $F: \mathbb{R}^n\to\mathbb{R}^n$

1D taylor expansion: 
$$f(x_{k+1}) = f(x_k + \Delta x_{k+1})$$
$$f(x_{k+1}) = f(x_k) + f'(x_k)\Delta x_{k+1} + \mathcal{O}(|\Delta x_{k+1}|^2)$$

n-dimensional case:
$$F(x_{k+1}) = F(x_k + \Delta x_{k+1})$$
$$F(x_{k+1}) = F(x_k) + F'(x_k)\Delta x_{k+1} + \mathcal{O}(|\Delta x_{k+1}|^2)$$

To go to the next step, we don't divide by $f'(x)$ like we do in the 1D case, but instead we solve the linear system
$$F'(x_k) \Delta x_{k+1} = - F(x)$$
where we know the first and last terms and want to find $\Delta x_{k+1}$

In the secant method, we just use $\hat{F}'(x)$ instead. This raises an extra problem: We need to find a good $\hat{F}'$

## Broyden's Secant Method

Secant equation: 
$$\hat{F}'_{k+1} \Delta x_k = \Delta F_k$$

We know $\Delta x_k$ and $\Delta F'_k$ from step k.

The problem: we have $n^2$ unknowns and 1 equation. What do we do?

We need to determine some $B_{k+1}$ that satisfies the secant equation.
1. "Absorb" $\Delta x_k$, and "Produce" $\Delta y_k$.
   $$\left( \Delta y_k \frac{\Delta x_k^T}{\Vert\Delta x_k \Vert^2}\right)\Delta x = \Delta y_k $$   
   The left part of the big bracket produces $\Delta y_k$, the right part eliminates $\Delta x_k$

2. Decide what $B_{k+1}$ does on the rest of $\mathbb{R}^n$. Anything that satisfies the secant equation will do 1., now we're into the arbitrary stuff. Broyden decides to let it act on the rest of the space in the same way as before (i.e, same as $B_k$). In math,
   $$ B_{k+1} = \Delta y_k \frac{\Delta x_k^T}{\Vert \Delta x_k\Vert^2} + B_k\left( I-P_{\Delta x_k}\right)$$
   $\left( I-P_{\Delta x_k}\right)$ is the rest of the space

### How to compute action on orthogonal component to $\Delta x$

$$P_{\Delta x_k} = \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k \Vert^2} $$
$$ \implies I = P_{\Delta x_k} = I - \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k \Vert^2} $$
In conclusion:
   $$ B_{k+1} = \Delta y_k \frac{\Delta x_k^T}{\Delta x_k^T \Delta x_k} + B_k - B_k\left( \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k \Vert^2} \right)$$
Be sceptical because it's fucking wrong, this guy's just btec Sneppen

It really looks like

$$ B_k + \frac{\Delta y_k \Delta x_k^T}{\Vert \Delta x_k\Vert^2} - B_k \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k\Vert^2} $$

Let's do the inverse instead

$$ B_{k+1}\Delta x_k = \Delta y_k $$
$$ \Delta x_k = B^{-1}_k\Delta y_k$$

There's nothing wrong with calculating the inverse here, we're using the same amount of information.

Let's build $\tilde{B}_{k+1}$ (which does the effect of the inverse, I think?) by swapping the roles of $\Delta x$ and $\Delta y$.

What do we start out with? We could start out with an approximation to the Jacobian, but there's no reason to: if we start out with the identity, we'll get a reasonable step that'll converge slowly, and as we gain information it'll get better and better.

## n-D to 1-D: Constrained equations on a curve

If you have a function $F: \mathbb{R}^m\to\mathbb{R}^n$ and $\gamma: \mathbb{R}\to\mathbb{R}^m$. ($\gamma$ is a parameterized curve in the highere dimensional space). 

Compose $F \circ \gamma: \mathbb{R}\to\mathbb{R}^n, \qquad F(\gamma(t))\in\mathbb{R}^n$.

Directional value (component along $\gamma(t)$): 
$$ F(\gamma(t))^T \gamma'(t) $$
i.e, dot product

**Special case**: straight line:
$$ \gamma(t) = x_0 + td$$
where $d$ is a vector with the direction of the line.
$$\gamma(t)' =d$$
$$\implies F_\gamma (t) = F(\gamma(t))^T d$$
	


# Non-linear Optimization

## When to use what:

### Questions that'll help decide on what to use
1. How slow are function evaluations (and gradients)?
2. How big is your space?
3. How ugly is your energy landscape? Convex, or many minima?

### Rules of thumb:
1. Fast function evaluations:
   1.  You can just take a linspace, evaluate the function at every point, and either use the minimum or feed that to a Newton Raphson method.  Both pretty and ugly energy landscapes works, it just changes the size of the linspace.
   2.  For medium dimensions (up to 100), you want to use BFGS if the energy landscape is simple. If it's complicated, you want to use BFGS + Exploration, as you need some way to escape local minima to find a global minima.
   3.  For high dimensions (up to 1M), use conjugated gradients. Takes longer to converge than BFGS, but you don't have to represent the high dimensional hessian matrix and so it works up to millions of dimensions. For a convoluted energy landscape, you want to use exploration as well.

2. Slow function evaluations
   1. Low to medium dimensions: If the energy landscape is simple, BFGS. Even if you have a complicated energy landscape, your search area is small enough that you can use BFGS with exploration.
   2. High dimensions: Simple energy landscape, use conjugate gradients. When it's expensive to evaluate the function, and you're in a high-dimensional complex landscape, the search space is too big for you to get anywhere. Here you need to think, and tailor yourr solution to fit your problem. Generally, you can try to use some sort of symmetry or structrue of your problem and then use a metaheuristic to guide your solutions

## Metaheuristics

_Algorithm:_ Computatoinal metho with guaranteed correct result after finite steps.

_Heuristic:_ The same as an algorithm, but with no guarantees.

An algorithm has 2 guarantees: it gives you a "correct" result (as per your definition of correct, we were using tolerance to decide that), and it happens after a finite number of steps. A heuristic doesn't guarantee either.

_Metaheuristic:_ A scheme for building heuristics. It's a framework where you have some overall structure (from your problem), then you take your metaheuristic scheme, tailor it to your problem, and then you produce a heuristic that you can run to get good answers for your problem

## Simulated Annealing:

Inspired by physical processes: annealing is when you let something cool slowly in order to get the correct hardened structure (in glass, metal, etc). 'Hardening' means that you find a position of atoms that minimizes the energy.

For minimizing $f:\mathbb{R}^n\to\mathbb{R}$ (maps from the high dimensional space to a scalar energy value):

* Maintain one state vector $x\in\mathbb{R}^n$
* Start with a high temperature $T_0$ ("high" is problem dependent).
* Gradually cool it down to 0 K

In each step:
1. Perturn $x$ by a random motion $\Delta x$
2. Let $\Delta E=(f(x+\Delta x)-f(x))\gamma$, where $\gamma$ is an optional energy unit.
3. Define a _transition probability_: $P(T, \Delta E) = e^{\Delta E/k_BT}$
4. Call a random number generator to get an $r\in[0,1]$ and accept new step if $P(T, \Delta E)\geq r$
5. Cool temperature: $T_{k+1} = \alpha T_k$

## Particle Swarm Optimizations

* Maintain multiple state vectors (a whole swarm of particles), $x_1,x_2,\dots,x_m \in \mathbb{R^n}$
  
In each step:
1. Pick a "free will" direction $\delta\in\mathbb{R}^n$ for each particle
2. Calculate $M\times M$ matrices
   1. The displacements $d:d_{ij}=x_j-x_i$
   2. Energy differences $F:F_{ij}=F(x_j)-F(x_i)$
3. Follow everyone who has a better solution: $f(x_j) < f(x_i)$, then $i$ follows $j$. We can program this as $\phi_{ij}=\frac12 \left(1-\text{sign}(F_{ij})\right)$. If $F_{ij}>0$, $\phi_{ij}=1$; $F <0$ then $\phi=0; F=0$ then $\phi=\frac12$
4. The direction of flight combines "Free will direction" $\delta_i$ with swarm movement: The direction is it's own independent movement (scaled by _independence factor_ $\iota$) and an _attraction factor_ $\alpha_{i,j}$ which is how much you want to follow the swarm:
$$\Delta x_i=\sum_{j=1}^M\alpha_{i,j}\phi_{i,j}d_{i,j} + \iota\delta_i$$

The attraction term ($\alpha$) is something that we must decide for ourselves. For example, $\alpha_{i,j} = \beta e^{-\gamma ||\delta_{i,j}||^2}$ (gaussian decline with distance: exponential will drop too quickly). You can also use a Levy distribution: small immediately around you, then spikes up close to you and declines slowly. You don't get pulled right on top of yourself. Even something like $\alpha_{ij}=\beta(F_{i,j})e^{-\gamma ||\delta_{i,j}||^2}$, where the attraction depends on the force of attraction.

Note: when programming: **don't** make loops over $i$ and $j$: do 2.2. like
```python
fs = array([f(x) for x in xs])
F = fs[:, np.newaxis] - fs[np.newaxis, :]
```
Similarly for 3.
```python
Phi =(1-np.sign(F))/2
```
And for 4.
```python
X += np.sum(alpha[:,:,np.newaxis]*phi[:,:,np.newaxis]*d, axis=1)
```
We use newaxis in order to math $\alpha$ and $\phi$ with $d$, which is a rank 3 tensor.

Note that $\delta$, the free will step, can also be tweaked. The simplest thing to do is to use brownian motion: you get something that explores a local area really well, but doesn't go very far out. If you use the Levy distribution instead of the gaussian, you get a combination of local exploration with (sometimes) big jumps to a new place.

## Genetic Algorithms

### How to start an evolution:

1. **Representation:** A genetic code
2. **Mating:** Processes for splitting and recombining genomes
3. **Selection Pressure:** Who, and whose offspring, make up the next generation?
4. **Mutation:** Radom perturbations (to get somewhere new)

Ideas $f:\mathbb{R}^{3N}\to\mathbb{R}$ representing $N$ particle position in space $\mathbb{R}^3$

	
	""")
with st.expander('Initial Value Problems for Ordinary Differential Equations', expanded=False):
	st.markdown('Linear Equations')
with st.expander('Partial Differential Equations	FFT and Spectral Methods', expanded=False):
	st.markdown('Linear Equations')




						
