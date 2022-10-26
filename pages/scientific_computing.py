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
	st.markdown('Linear Equations')
with st.expander('Eigensystems', expanded=False):
	st.markdown('''
* Gershgorin centers
* rayleigh quotient
* power iterate (gives us the greatest eigenvalue)
* rayleigh iterate''')
with st.expander('Nonlinear Equations	Optimization', expanded=False):
	st.markdown('Linear Equations')
with st.expander('Initial Value Problems for Ordinary Differential Equations', expanded=False):
	st.markdown('Linear Equations')
with st.expander('Partial Differential Equations	FFT and Spectral Methods', expanded=False):
	st.markdown('Linear Equations')




						
