

# Header 1
Sometimes problems are non-linear -  this makes everything ugly and difficult...

We can use monte carlo sampling for example for find the mean.


we want to check samples whihch are taken from the pdf f(x).


Then we take a sample, and we use rejection sampling (simple) to choose whether or not to keep it....

$$
	p_\text{accept} = \frac{p(\mathbf{x}_\text{cand})}{M}
$$
in which M is some arbitrary, large value. Ie, $M\geq \text{max}_\mathbf{x}(p(\mathbf{x}))$

# Header 2
we could make this algorithm more sofiticated (rejection sampling) by adding a term $q$. 

$$
	p_{accept} = \frac{p(\mathbf{x}_\text{cand})}{M q(\mathbf{x}_\text{cand})}
$$


MCMC
* propose jump chosen from a probability distribution

accept $\mathbf{x}_j$ only with probability

$$
	p_\text{accept} = \text{min}\left( 1, \frac{p(\mathbf{x}_i)}{p(\mathbf{x}_j)}\right)
$$
otherwise repeat $\mathbf{x}_j$




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



> Relative entropy is translation invaritant: Also called the Kullback Leibler distance between $p(x)$ and $q(x)$, i.e., the different between two probability densities.
$$
    D(p||q) = \sum_x p(x)\log\frac{p(x)}{q(x)}
$$
which has the properties: 

$$
    D(p||q) \geq 0\\
    D(p||p) = 0 \\
$$
