
## Header 1

Any stable physical/chemical system can be described with an eigensystem. Wavefunctions in quantum mechanics are a typical example.
$$
Ax = \lambda x
$$
$A$ is the operator (mapping $\mathbb{C}^n \rightarrow \mathbb{C}^n$), $x$ an eigevector, $\lambda$ an eigenvalue. The question is, **How do we obtain the eigen-vectors and -values?**

## Header 2

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

#### Power method algorithm

```python
def PowerIterate(A, x_0):
    x = x_0
    while (not_converged(A, x)):
        x = A @ x
        x /= np.linalg.norm(x)
    return x
```

You can check for convergence by seeing if it's an eigenvector. Either check if $Ax$ is parallel to $x$, or use the Rayleigh equation (which we'll cover in lecture 6).

### Something else he's squeezed in

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

## Gershgorin centers
### Gershgorin centers
We can appoximate the eigenvalues using the Gershgorin center method:

## gershgorin
def gershgorin(K, _sort=True):
    # Starting with row one, we take the element on the diagonal, $a_{ii}$ as the center for the disc. # wikipedia
    centers = np.array([K[i,i] for i in range(len(K))])

    # We then take the remaining elements in the row and apply the formula:
    # $$\sum _{{j\neq i}}|a_{{ij}}|=R_{i}$$
    radii = np.array([np.sum(np.hstack( [np.abs(K[i,:i]), np.abs(K[i,i+1:])] )) for i in range(len(K))])
    
    if _sort:  # sorting
        sort_index = np.argsort(centers)[::-1]
        centers = centers[sort_index]; radii = radii[sort_index]
    return centers, radii


## rayleigh iterate


#### Rayleigh quotient
```python
def rayleigh_qt(A,x=None):
    return x.T @ A @ x / (x.T @ x)
```

#### Rayleigh iterate
```python
def rayleigh_iterate(A,x0, shift0, tol=0.001):
    norm = lambda v : np.max(abs(v))
    
    A = A.copy()
    
    ys = [] ; xs = [x0] ; cs = []
    lambda_ = 420 ; lambda_prev = -300
    A_less_eig = A - shift0*np.eye(len(A))
    i = 0
    while abs(lambda_-lambda_prev) > tol:
        lambda_prev = lambda_
        ys.append(solve_lin_eq(A_less_eig, xs[i]))
        cs.append( ( ys[i] @ xs[i] ) / (xs[i] @ xs[i]) )
        xs.append( ys[i]/norm(ys[i]) )
        
        lambda_ = 1/cs[i] + shift0
        i +=1
        
        
    return lambda_, xs[-1], i
```
