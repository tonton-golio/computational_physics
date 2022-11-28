
# description
* Nov 28: Probability Density Functions (PDF) especially Binomial, Poisson and Gaussian.
* Nov 29: Principle of maximum likelihood and fitting (which is an art!).
* Dec 2: 8:15 - Group A: Project (for Wednesday the 14th of December) doing experiments in First Lab. 9:15 - Group B: Systematic Uncertainties and analysis of "Table Measurement data" Discussion of real data analysis (usual rooms).


# Header 1
### Probability density functions (PDFs)
*a function of a continuous random variable, whose integral across an interval gives the probability that the value of the variable lies within the same interval.*

$$
     f_X(x) = \frac{1}{\Delta x}\int_{x0}^{x0+\Delta x} f(x) dx
$$

When fitting with PDFs, we should consider the error stemming from the bin-widths 🥸

We may also consider the cumulative distribution function: just take the integral from $-\infty$,
$$
    F_X(x) = \int_{-\infty}^{x0} f(x) dx.
$$




# Distributions
### Binomial
N trials, p chance of succes, how many successes should you expect
$$
\begin{align*}
     f(n;N,p) &= \frac{N!}{n!(N-n)!}p^n(1-p)^{N-n}\\
     \left<f(n;N,p)\right> &= Np \\
     \sigma^2 &= Np(1-p)
\end{align*}
$$


Holy crap, its discrete 🤯

### Poisson
if $N\rightarrow \infty$ and $p\rightarrow 0$, but $Np\rightarrow\lambda$ i.e., some finite number. Then a binomioal approaches a Poisson:
$$
     f(n, \lambda) = \frac{\lambda^n}{n!}e^{-\lambda}
$$

**Exmaple**: A whole lot of danes go into traffic every day ($N\rightarrow\infty$), the possitilibty of being killed in traffic is tiny ($p\rightarrow 0$) --> but some people do get killed in trafic every year $\lambda\neq 0$;(

The Poisson hase mean and variace $\lambda$. The error on a Poisson number is the square-root of that number.

**A useful case**: the error to assign a bin in a histogram if there is reasonable statistics ($N \rightarrow 5-20$) in each bin. If there are low statistics in a bin, we cannot make the gaussian approximation!?!?!?



The sum of Poisson's is a Poisson: $\lambda = \lambda_a + \lambda_b$


If $\lambda \rightarrow \infty$, the Poisson becomes gaussian ($\infty\approx20$) [SHOW THIS PREASE]


### Gaussian
$$
     \frac{1}{2\pi}\ldots
$$


### Student's t-distribution
Good for low statistics, because it uses the approximations of $\mu$ and $\sigma$. Other than that, it is similar to the Gaussian.

When number of samples $\rightarrow\infty$ this becomes the Gaussian.
$$
     \ldots
$$


# Header 2
### Principle of maximum likelihood and fitting
$$
     likelihood = math
$$

we want to maximize this term ^^