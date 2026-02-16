# Probability Density Functions

# description
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
The normal normal distribution;
$$
     f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
$$
This is awesome, because given large number of samples we usually always observe this and its easy to work with.


### Student's t-distribution
Good for low statistics, because it uses the approximations of $\mu$ and $\sigma$. Other than that, it is similar to the Gaussian.

When number of samples $\rightarrow\infty$ this becomes the Gaussian.
$$
     \ldots
$$

### blank

# maximum likelihood
### Maximum likelihood estimation

We are on the look out for the maximum likelihood, $\mathcal{L}$, given the observed data, assuming a normal distribution of the sample.
$$
     \mathcal{L}(\theta) = \prod_i f(x_i, \theta) dx_i
$$
We are prone to rounding errors when multiplying a bunch of small numebrs together; so lets instead sum the logs. Assuming that things are Gaussian;
$$
     \log{\mathcal{L}} = \sum N  \exp\left(
          \frac{x-\mu}{r}
     \right)^2 +c\\
     -2\ln(\mathcal{L}) = \chi^2 + c
$$

* It consistent
* Asymptotically normal (converges with Gaussian errors.)
* Efficient (reaches the Minimum variance bound (MVB, Cramer-Rao) for large N)




The way we figure this out is;
* take a normal dist, and move the center to where it fits best
* next up; $\sigma$.



# maximum likelihood 2
Notice there are different methods depending on whether we are using individual data-points or the bins and counts obtained from `np.histogram`.

Consider doing the binned approach if we are working with a really large sample (because the likelihood calculation is $O(n^2)$). 


**Next up we have to test our fit: sample new number from the Gaussian with the found parameters.**

##### The likelihood ratio test
if we have two different hypothesis
$$
          d 
          = -2\ln\frac{\mathcal{L}_\text{null}}{\mathcal{L}_\text{alt.}}
          = -2\ln(\mathcal{L}_\text{null}) + 2\ln(\mathcal{L}_\text{alt.})
$$

Consider the DOF for each hypothesis. Greater DOF with nessecarily yield likelihood. 