
# title
Introduction, General Concepts, ChiSquare Method

# description
##### Introduction, General Concepts, ChiSquare Method

* Nov 21: Introduction to course and overview of curriculum. Mean and Standard Deviation. Correlations. Significant digits. Central limit theorem. 
* Nov 22: Error propagation (which is a science!). Estimate g measurement uncertainties.
* Nov 25: ChiSquare method, evaluation, and test. Formation of Project groups.


# Header 1
*Don't worry too much about the theory, this is an **applied** course*.
*We'll try to build a generally applicable `code repo`*.

* The magic happens outside your comfort zone.

# Mean
Mean is a metric telling us about bulk magnitude-tendency of data. 

# Geometric mean
root of the product,
$$
     \bar{x}_\text{geo} = \left( \prod_i^n x_i\right)^{1/n}
$$
is equivalent to the arithmetic mean in logscale
$$
     \exp\left(\frac{1}{n}\sum_{i}^n\ln x_i \right)
$$
# Geometric mean code
```python
def geometric():
     return np.prod(arr)**(1/n)
```

# Arithmetic mean
Equally weighted center,
$$
     \hat{\mu} = \bar{x} = \left< x \right> = \frac{1}{N}\sum_i^N x_i.
$$

*can we define this differently, i.e. in a more intuitive manner?* The arithmetic mean is that in which:

$$
     x_i = \bar{x} + \delta_i.
$$
# Arithmetic mean code
```python
def arithmetic():
     return np.sum(arr) / n

def arithmetic():
     return np.mean(arr)
```

# Median
The counting center point of the data
# Median code
```python
def median():
     return arr[n/2] if n%2==0 else arr[n//2+1]
```

# Mode
The most typical data point
# Mode code
```python
def mode():
     v2c = value_counts = {}
     for i in arr:
          v2c[i] = v2c[i] + 1 if i in v2c else 1
     
     c2v = dict((val, key) for (key, val) in dict.items())
     return c2v[max(c2v.keys())]
```

# Harmonic
Harmony
# Harmonic code
```python
def harmonic():
     return (np.sum( arr**(-1) ) / n)**(-1)
```

# Truncated
Truncated
# Truncated code
```python
def truncated():
     arr = arr[truncate:-truncate]
     return arithmetic(arr)
```



# STD

Standard deviation is a measure of how much datapoint deviates from the dataset mean, ``np.std(x)``.

$$
     \hat{\sigma} = \sqrt{\frac{1}{N}\sum_i^N ( \left< x \right> -x_i)^2}
$$

In estimating this, we assume that we know the real mean. But in reality we dont really know this so we use:

$$
     \hat{\sigma} \approx \sqrt{
     \frac{1}{N-1}\sum_i(x_i-\bar{x})^2
     }
$$
we subtract 1, because something something degrees of freedom...

Lets compare these:

# Weighted mean

###### Weighted mean
How to average data which has different uncertainties and what is the uncertainty on the average?

$$
    \begin{align*}
        \hat{\mu} = \frac{\sum x_i / \sigma_i^2}{\sum 1 / \sigma_i^2}
        , &&
        \hat{\sigma_\mu} = \sqrt{\frac{1}{\sum 1/\sigma_i^2}}
    \end{align*}
$$
Uncertainty descreases with the squares of the number of sampels;
$$
    \hat{\sigma_\mu} = \hat{\mu}/\sqrt{N}.
$$


# Correlation
Correlation speaks to whether a feature varies in concordance with another.

From the variance we can obtain the covariance; 
$$
     \begin{align*}
     V = \sigma^2 &= \frac{1}{N}\sum_i^N(x_i-\mu)^2 = E[(x-\mu)^2] = E[x^2] - \mu^2\\
     &\Rightarrow \\ 
     V_{xy} &= E[(x_i-\mu_x)(y_i-\mu_y)]=
     \begin{bmatrix}
          \sigma_{11}^2 & \sigma_{12}^2 & \ldots\\
          \sigma_{21}^2 & \ldots & \ldots
     \end{bmatrix}
     \end{align*}
$$

Normalizing by the widths gives the Pearson's (linear) correlation coefficient:

$$
\begin{align*}
     \rho_{xy} = \frac{V_{xy}}{\sigma_x\sigma_y} , && \text{i.e., } -1 < \rho_{xy} < 1 
\end{align*}
$$

Do bare in mind that we may get zero, but this just tells us that the correlation is not linear, so remember to plot 😉

##### Rank correlation
**Test for non-linear correlation.**

Rank correlation compares the ranking between two sets, i.e., 
if the lowest x is also the lowest y, and so on, we get a rank-correlation of 1 😁 *Spearman’s $\rho$* and *Kendall’s $\tau$* are the most typical rank correlations.

$$
     \rho = 1 - 6\sum_i \frac{(r_i - s_i)^2}{n^3-n}
$$

*Kendall’s $\tau$* compares the number of concorant pairs to discoratant pair of data-points [[wiki]](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient).


##### Non-linear correlation-measures include

* Maximal Information Coefficient (MIC)
* Mutual Information (MI)
* Distance Correlation (DC)

*see wikipedia*


# Central limit theorem
##### *law of large numbers*

The central limit theorem answers the question: *why do statistics in the limit of large N tend toward a Gaussian?*

The sum of N independent continous random variables $x_i$ with means $\mu_i$ and variances $\sigma_i^2$ becouse a Gaussian random varianble with mean $\mu=\sum_u\mu_i$ and variance $\sigma^2 = \sim_i \sigma_i^2$ in the limit than N approaches infinity.

*The distribution of numbers samples from a variety of distrutions, is gaussian given that they share variance and mean.*

# Central limit theorem 2
###### Combining samples from different distributions
if we concatenate data from different distributions, the mean will be Gaussian.

# Central limit theorem 3
For this to work, the std of the different distributions must be similar. Thus we have to scale the uniform by a factor $\sqrt{12}$, and we have to truncate the very large values in the chauchy distribution.

In summary: **The central limit theorem ensures that uncertanties are gaussian 🔥**


# Error propagation
### Error propagation
If we have a function $y(x_i)$ and we know the uncertainty on $\sigma(x_i)$, how do we find $\sigma(y(x_i))$?. Well obviously it depends on the gradient of $y$ with respect to $x$.

A simple way of doing this:
$$
    \sigma(y(x_i)) = \frac{\partial y}{\partial x}\sigma(x_i)
$$
If the function $y$ is smooth around $x_i$, this is fine, but if we have a crazy looking function, we should be careful. (The slope should be relatively constant over the uncertainty in $x_i$.)


If we Taylor expand:
$$
    y(\bar{x}) = \ldots\\
$$

something something, this lets us expand to further dimensions:

$$
     \sigma_y^2 = \sum^n_{i,j}\left[
     \frac{\partial y}{\partial x_i}
     \frac{\partial y}{\partial x_j}
     \right]_{x=y}V_{i,j}
$$
if there are no correlations, only the diagonal (individual erros) enter!
This lets us choose wisely which parameter we should work hard to minimize error on.

# Error propagation add
###### Addition

$$
    y = x_1 + x_2\\
    \sigma_y^2 = \sigma_{x1}^2 + \sigma_{x2}^2 + 2V_{x1, x2}
$$

# Error propagation mul
###### Multiplication

$$
    y = x_1x_2\\
    \sigma_y^2 = (x_2\sigma_{x1})^2 + (x_1\sigma_{x2})^2 + 2x_1x_2V_{x1, x2}
$$


divide by $x^2$ to get the relative terms... 

By using negative error correlations, we can cancel out errors, think: Harrisons girdiron pendulum, and the prediction of Neptune!





# Demo
#####  Simulating error propagation
Choose random input, $x_i$, and record output, $y$. If $y(x)$ is not smooth, this will not yield Gaussian distributed $y$, even with gaussian error in $x_i$.


How do we quantify the error stemming from limited random sampling? -- *don't worry about it.*



# Error propagation 2
##### Analytical solution
... sympy

# Estimating uncertainties
### Estimating uncertainties


# ChiSquare method, evaluation, and test
### ChiSquare method, evaluation, and test

A linear fit, can be done analytically.

Least squares -> doesnt include uncertainties
chi-square -> includes uncertainties.

$$
     \chi^2(\theta) = \sum_i^N \frac{(y_i - f(x_i,\theta))^2}{\sigma_i^2}
$$

So its like least-sqares, but we scale by the uncertainty.

*Why don't we normalize by the number of samples, $N$*


To compare chi-square fits,
* number of degrees of freedom = samples - fitparameters.
$$
     N_\text{Dof} = N_\text{samples} - N_\text{fit parameters}
$$
* take the probability of the data with that Ndof, i.e., the $p$-value.
$$
     \text{Prob}(\chi^2 = 65.7, Ndof=42) = 0.011
$$

With large errors, quality of fits with different models become the same. With small errors, there is a huge discrepancy in the fit-quality.


When we do this fitting: we get back the optimal parameters, their uncertainties and the covariance of the parameters.


Notice: the weighted mean is just a special case of chi-squared, i.e., the value of a fit with just a constant.


(**Show this with a plot**)



If we plot the distribution of the errors as $\frac{y_i-f(x_i, \theta)}{\sigma_i}$, they should be Gaussian.


```python
chi2_prob = stats.chi2.sf(chi2_calue, N_dof)
```

Now we have a way to evaluate our uncertainty-estimates.




###### Chi2 for binned data
based on Poisson statistics
$$
     \chi^2 = \sum_{i\in \text{bin}} \frac{(O_i-E_i)^2}{E_i}
$$
The problem with this, is where to truncate?


The alternative:

use counts in bins! But then what about empty bins!!! Minuit does it this way but all methods have their draw-backs.


###### Why the chi2 is (near) magic
(Inser chi2_minimization_animation_original)


Nicely behaving fits have parabolic minima. --> The chi2 miracle


## Header to call below

minuit can fin the uncertainty on chi2 even though it does not know that we are fitting with a Gaussian... HOW?!?!?!?! The way it does this is by looking at the curvature of the parabolic...



### Question: How do we deal with uncertainties in $x$?
Fit without, and then: 
$$
     \sigma_{yi}^{new} = \sqrt{\sigma_{yi}^2 + \left( \frac{\partial y}{\partial x}|_{xi} \sigma_{xi} \right)^2}
$$
Its iterative, but it converges FaST


### Reporting errors 

do this:r
$$
    0.24 + 0.05_\text{stat} + 0.07_\text{systematic} 
$$


# Links
### Links
check out [guess the correlation](https://guessthecorrelation.com)

