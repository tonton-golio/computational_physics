
# Header 1

###### Monte Carlo
> We wanna sample our space, typically using Monte Carlo... This is done to save compute. We will save a point depending on the value evaluated at that specific parameter configuration.

> If we wanna locate the minimum on an array of size $n=8$, we must sub-divide our space and ask "which contains the extrema". The number of questions meccesitated in obtaining the extrema is $\log_2(n)$.

> in the case where we dont have equiprobable events, we use $H(p_k) = \sum_k p_k \log(\frac{1}{p})$ to define the entropy.

> This case above is only valid for the discrete case however... How do we expand to the continous domain?!?!?!?!!?

> We do this:
$$
    H(f) = -\int_{-\infty}^\infty f(x)\log_2(f(x))dx
$$
> it not really entropy because we made it by subtraction of two functions. Thus we call it differential entropy.

> @Horiike --> I love this way of typing 🤍


> Relative entropy is translation invaritant: Also called the Kullback Leibler distance between $p(x)$ and $q(x)$, i.e., the different between two probability densities.
$$
    D(p||q) = \sum_x p(x)\log\frac{p(x)}{q(x)}
$$
which has the properties: 

$$
    D(p||q) \geq 0\\
    D(p||p) = 0 \\
$$




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



> we may deal with cases which are either under- or over-determined.