

# description
(Introduction, General Concepts, ChiSquare Method):
Nov 21: 8:15-10:00: Introduction to course and overview of curriculum.
Mean and Standard Deviation. Correlations. Significant digits. Central limit theorem. (12-13 Measuring in Aud. A!)
Nov 22: Error propagation (which is a science!). Estimate g measurement uncertainties.
Nov 25: ChiSquare method, evaluation, and test. Formation of Project groups.


# Header 1
*Don't worry too much about the theory, this is an **applied** course* 😗
The central goal is to *build a generally applicable `code repo`*.


# Mean
We have a bunch (6) of different means:

# mean 1
* Geometric mean
* Arithmetic mean
* Median - *half below, half above*

# mean 2
* Mode
* Harmonic mean
* Truncated mean

# means
##### Geometric mean

mean (*average*) is the geometric midpoint of a data-set, ``np.mean(x)``.

$$
     \hat{\mu} = \bar{x} = \left< x \right> = \frac{1}{N}\sum_i^N x_i
$$

*can we define this differently, i.e. in a more intuitive manner?*

$$
     x_i = \bar{x} + \delta_i\\
$$
the deviations from the mean of every data-point, must neccessarily sum to null.

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

#### Weighted mean
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


# Correlations
Speaks to whether a feature varies in concordance with another.

With variance defined as  
$$
     V = \sigma^2 = \frac{1}{N}\sum_i^N(x_i-\mu)^2 = E[(x-\mu)^2],
$$

we may obtain the covariance:

$$
     V_{xy} = E[(x_i-\mu_x)(y_i-\mu_y)].
$$

Normalizing by the widths gives the Pearson's linear correlation coefficient:

$$
     \rho_{xy} = \ldots
$$

Do bare in mind that we may get zero, but this just tells us that the correlation is not linear, so remember to plot 😉


The covariance matrix looks like:

$$
     V_{xy} = \begin{bmatrix}
     \sigma_{11}^2 & \sigma_{12}^2 & \ldots\\
     \sigma_{21}^2 & \ldots & \ldots
     \end{bmatrix}
$$


### Rank correlation

if the lowest x is also the lowest y, and so on, we get a rank-correlation of 1 😁

Non-linear correlations include

* maximal intfortimaton
* mutual ...
* 3rd one >?? (slide 15)


# Central limit theorem
### Central limit theorem
**law of large numbers...**

The central limit theorem answers the question: *why do statistics in the limit of large N tend toward a Gaussian?*

**The sum of N independent continous random variables $x_i$ with means $\mu_i$ and variances $\sigma_i^2$ becouse a Gaussian random varianble with mean $\mu=\sum_u\mu_i$ and variance $\sigma^2 = \sim_i \sigma_i^2$ in the limit than N approaches infinity.**


The reason a small system shows a soft transition has something to do with the central limit theorem.


Standard deviations of the gaussian distribution

(table plz)
1 std = 68%
2 std = 95%
3 std = 99.7%
5 std = 99.99995%

(show a plot of this)

In summary: **The central limit theorem ensures that uncertanties are gaussian 🔥**

# Header 4


### Error propagation
hmm, how do we propagate these errors? no se?

If we have a function $y(x_i)$ and we know the uncertainty on $\sigma(x_i) = 0.8$, how do we find $\sigma(y(x_i))$?. Well obviously it depends on the gradient of $y$ with respect to $x$.

A simple way of doing this:
$$
    \sigma(y(x_i)) = \frac{\partial y}{\partial x}\sigma(x_i)
$$
If the function $y$, is smooth around $x_i$, this is fine, but if we have a crazy looking function, we should be careful. (The slope should be relatively constant over the uncertainty in $x_i$.)



If we Taylor expand:
$$
    y(\bar{x}) = \ldots\\
$$

bla bla bla, this lets us expand to further dimensions:

$$
     \sigma_y^2 = \sum^n_{i,j}\left[
     \frac{\partial y}{\partial x_i}
     \frac{\partial y}{\partial x_j}
     \right]_{x=y}V_{i,j}
$$
if there are no correlations, only the diagonal (individual erros) enter!
This lets us choose wisely which parameter we should work hard to minimize error on.


#### Addition

$$
    y = x_1 + x_2\\
    \sigma_y^2 = \sigma_{x1}^2 + \sigma_{x2}^2 + 2V_{x1, x2}
$$


#### Multiplication

$$
    y = x_1x_2\\
    \sigma_y^2 = (x_2\sigma_{x1})^2 + (x_1\sigma_{x2})^2 + 2x_1x_2V_{x1, x2}
$$
divide by $x^2$ to get the relative terms...



By using negative error correlations, we can cancel out errors, think: Harrisons girdiron pendulum


Also prediction of Neptune!!!!!


#### Simulating error propagation
choose random input, $x_i$, and record output, $y$. If $y(x)$ is not smooth, this will not yield Gaussian distributed $y$, even with gaussian error in $x_i$.


how do we quantify the error stemming from limited random sampling




### Reporting errors 

do this:
$$
    0.24 + 0.05_\text{stat} + 0.07_\text{systematic} 
$$



### Estimating uncertainties





###  ChiSquare method, evaluation, and test


# Links
## Liinks
check out [guess the correlation](https://guessthecorrelation.com)

