
### Hurst exponent
The Hurst exponent is a measure of long-term memory of timeseries.
$$
    \mathbb {E} \left[{\frac {R(n)}{S(n)}}\right]=Cn^{H}{\text{  as }}n\to \infty \,,
$$
where;

* $R(n)$ is the range of the first $n$ cumulative deviations from the mean
* $S(n)$ is the series (sum) of the first n standard deviations
* $\mathbb {E} \left[x\right]\,$ is the expected value
* $n$ is the time span of the observation (number of data points in a time series)
* $C$ is a constant.

$H$ ranges between 0 and 1, with higher values indicating less volatility/roughness. For self-similar time-series, $H$ is directly related to fractal dimension, $D=2-H$.


### Fear-factor model
If we assume that stock prices move in a Brownian fashion, we may let the $q$ describe the probability of the stock moving up. Naturally $1-q$ is the probability of the stock moving down.

We let $p$ be the probability of a *collective fear event* which causes all stocks to move down. We thus have $(1-p)q$ probability of the stock increasing and $p+ (1-p)(1-q)$ probability of the opposite.

A neutral random walk (Brownian) requires that these probabilities are equal;
$$
    (1-p)q = p+ (1-p)(1-q) \Rightarrow q = \frac{1}{2(1-p)}
$$

### Bet-Hedghing Model
Bet-hedghing ...
an important parameter in bet-hedghing models, is the noise size