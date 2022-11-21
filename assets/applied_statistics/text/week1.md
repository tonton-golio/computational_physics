

# description
(Introduction, General Concepts, ChiSquare Method):
Nov 21: 8:15-10:00: Introduction to course and overview of curriculum.
Mean and Standard Deviation. Correlations. Significant digits. Central limit theorem. (12-13 Measuring in Aud. A!)
Nov 22: Error propagation (which is a science!). Estimate g measurement uncertainties.
Nov 25: ChiSquare method, evaluation, and test. Formation of Project groups.



# Header 1
*Don't worry too much about the theory, this is an **applied** course* 😗


In this course, a central goal is to *build a generally applicable repository*.



### mean and standard deviation
#### MEAN
We have a bunch (6) of different means:

* Geometric mean
* arithmetic mean
* median - *half below, half above*
* 
* 
* 

##### Geometric mean


mean (*average*) is the geometric midpoint of a data-set. 

``np.mean(x)``

$$
     \hat{\mu} = \bar{x} = \left< x \right> = \frac{1}{N}\sum_i^N x_i
$$

*can we define this differently, i.e. in a more intuitive manner?*

$$
     x_i = \bar{x} + \delta_i\\
$$
the deviations from the mean of every data-point, must neccessarily sum to null.

#### STD

Standard deviation is a measure of how much datapoint deviates from the dataset mean.

``np.std(x)``


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

Lets compare these...

# Header 2
Weighted mean:
$$
     \hat{\mu} = \frac{\ldots}{\ldots}
$$

check out (guess the correlation)[https://guessthecorrelation.com]



Variance = 
$$
     V = \sigma^2 = \frac{1}{N}\sum_i^N(x_i-\mu)^2 = E[(x-\mu)^2]
$$

this yields the covariance:

$$
     V_{xy} = E[(x_i-\mu_x)(y_i-\mu_y)]
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


# Header 3

### Correlations
speaks to whether a feature varies in concordance with another.

iris = sns.load_dataset('iris')
corr = pd.corr(iris)

$$
     math
$$

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
hmm, how do we propagate these errors?



### Estimating uncertainties





###  ChiSquare method, evaluation, and test


