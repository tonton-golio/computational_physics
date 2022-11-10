

### Brownian Motion
Let $S_t$ be the closing price for a stock, and $x_t=\log(S_t)$. The variance for a lag $\tau$ is given by:
$$
    \text{var}(\tau) = \left<
        (x_{t+\tau}-x_t)^2
    \right>
$$
For geometric Brownian motion (random walk), the variance at an arbitrary lag is proportional to that lag, $\text{var}(\tau)\propto \tau$ .

### Brownian Motion 2
Notice, the plot on the right indicates that we are not very far from Brownian motion.

### Hurst exponent
The Hurst exponent is a measure of long-term memory of timeseries.

$$
\begin{align*}
    \tau
    &=
    Cn^{H}{\text{  as }}n\to \infty \,
    \\
    &\Rightarrow
    H =  \frac{\log(Cn)}{\log{\tau}}
\end{align*}
$$
where; $n$ is the number of data points in a time series, and  $C$ is a constant.


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


### Resources
| name/link      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |