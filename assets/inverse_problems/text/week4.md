

# Header 1
omg sometimes problems are non-linear.... this makes everything ugly and difficult...

We can use monte carlo sampling for example for find the mean.


we want to check samples whihch are taken from the pdf f(x).


Then we take a sample, and we use rejection sampling (simple) to choose whether or not to keep it....

$$
	p_\text{accept} = \frac{p(\mathbf{x}_\text{cand})}{M}
$$
in which M is some arbitrary, large value. Ie, $M\geq \text{max}_\mathbf{x}(p(\mathbf{x}))$


### Sampling example
if we throw darts at a unit-hypercube, we are to hit inside the unit-hypersphere $\frac{\pi}{2^d}$ of the time.