
# 1
## Least-squares - Tikonov regularization
Least square in the case of inverse problems is quite straight forward:
minmize the norm of $g(m)-d$, in which $g$ is the function, $m$ are the parameters and $d$ is the observed data.

This simple approach is however prone to overfitting.

We thus have Tikhonov regularization, which for linear regression can be represented by the following cost function:
$$
\mathcal{L} = ||d - g(m)||^2 + \epsilon \times ||m||^2
$$
Where:
$\epsilon$ is the regularization parameter, a scalar that determines the strength of the regularization term.

The cost function is minimized over the parameters in order to obtain the optimal solution of Ridge Regression.

The above cost function can be minimized by solving the following regularized normal equations:

$$
    m = (X^TX + \epsilon I)^-1 \cdot X^TY
$$
Where X is the training data input, Y is the training data output, and I is the identity matrix.




### Least squares and Tikhonov regularization
If the relation is linear;
$$
    \mathbf{d} = g(\mathbf{m}) \Rightarrow
    \mathbf{d} = \mathbf{G}\mathbf{m},
$$
the least squares problem is to minimize 
$$
    E(\mathbf{m}) = ||\mathbf{d}-\mathbf{Gm}||^2
$$

This can be solved analytically, a solution vector $\mathbf{\hat{m}}$ satisfies
$$
    \forall j: \frac{\partial E}{\partial \hat{m}_j} = 0
$$


> OMG, look out for noise!!!

$$
    \mathbf{\hat{m}} = \mathbf{G}^T (\mathbf{G}\mathbf{G}^T)^{-1}\mathbf{d}
$$



> we may deal with cases which are either under- or over-determined. 
* **Over determined** -> 1 solution but its not exact
* **under determined** -> many solutions...


To get around this over- or under-determination, *use*:

$$
     E(\mathbf{m}) = ||\mathbf{d}-\mathbf{Gm}||^2 + \epsilon^2||\mathbf{m}||^2
$$
which yields a minimum given by:
$$
    \mathbf{\hat{m}} = \mathbf{G}^T (\mathbf{G}\mathbf{G}^T + \epsilon^2\mathbf{I})^{-1}\mathbf{d}\\
    \Rightarrow\\
    \mathbf{\hat{m}} = (\mathbf{G}\mathbf{G}^T + \epsilon^2\mathbf{I})^{-1}\mathbf{G}^T \mathbf{d}
$$



##### How do we know if we have a mixed-determined problem?

> Overdetermined if we have more samples, $N$, than we have parameters, $m$. i.e., if the matrix rank is smaller than the number of rows.

> Underdetermine if we have less samples, $N$, than we have parameters, $m$. i.e., if the matrix rank is smaller than the number of columns.


If the rank is smaller than $N$ and $m$, then we have a mixed-determined system.


###### Summary of the key-points

Almost all linear inverse problems we will face are mixed-determined!?!?!?! So we have to use Tikinov's formula ❤️



# Header 2
## Lecture notes
#### Ocean floor magnetization

$$
    d_i = \int_{-\infty}^\infty g_i(x)m(x)dx\\
$$

discretize
$$
    \mathbf{m} = (m(x_1),m(x_2),\ldots)\\
    \Rightarrow\\
    d_i \approx \sum_{k=1}^M g_i(x_k)m_k\Delta x
$$
Matrix formulation
$$
    G_{ij} = g_i(x_j) \Rightarrow \mathbf{d} = \mathbf{Gm}
$$


Now we just solve ;
$$
    ||\mathbf{d}_\text{obs} - \mathbf{G\hat{m}}||^2\approx N\sigma^2.
$$

(i.e., the difference between the observed data and estimated results should be on the order of the uncertainty in the measurements.)

To do so, move all terms on side and minimize 💪



##### Weighting data and model with their standard deviations

define
$$
    \bar{\mathbf{d}} = \mathbf{Vd}
$$
where
$$
    \mathbf{V}^T\mathbf{V} = \mathbf{C}_D^{-1}
$$
and
$$
    \bar{\mathbf{m}} = \mathbf{Wm}
$$
where
$$
    \mathbf{W}^T\mathbf{W} = \mathbf{C}_M^{-1}
$$

theis means
$$
\begin{align*}
    \mathbf{d}=\mathbf{Gm} &\rightarrow \mathbf{Vd} = \mathbf{VGm}\\
                           &\rightarrow \mathbf{Vd} = \mathbf{VGW}^{-1}\mathbf{Wm}\\
                           &\rightarrow \mathbf{\bar{d}} = \mathbf{VGW}^{-1}
\end{align*}
$$
where
$$
    \bar{\mathbf{G}}= \mathbf{VGW}^{-1}
$$


Lets put this into the Tikhonov formula:
$$
    ...
$$



## Transformed problem
To obtain the transformed problem just use the singular value decomposition (SVD). *Small singular value, give strong noise amplification.*

Sorry I didn't listen very well... (but, fear not: I'll be back with the fine grit sandpaper)


























