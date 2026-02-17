# Econophysics

## Statistical mechanics of financial markets

**Econophysics** applies methods from statistical mechanics and complex systems to understand financial markets. The central observation is that financial time series share many statistical properties with physical systems: heavy-tailed distributions, long-range correlations, and scaling behavior.

## Brownian motion and stock prices

The simplest model treats log-returns as a random walk. Let $S_t$ be the closing price and $x_t = \log(S_t)$. The variance at lag $\tau$ is:

$$
\text{Var}(\tau) = \langle (x_{t+\tau} - x_t)^2 \rangle.
$$

For **geometric Brownian motion** (the foundation of the Black-Scholes model), variance grows linearly with lag:

$$
\text{Var}(\tau) \propto \tau.
$$

In practice, real stock returns deviate from pure Brownian motion: they exhibit heavier tails (large moves are more frequent than a Gaussian predicts) and volatility clustering (large changes tend to follow large changes).

[[simulation stock-variance]]

## The Hurst exponent

The **Hurst exponent** $H$ measures long-range dependence in time series. For the rescaled range $R/S$ of a time series with $n$ data points:

$$
\frac{R}{S} \sim C \, n^H \quad \text{as } n \to \infty,
$$

where $C$ is a constant.

The value of $H$ reveals the nature of correlations:

- $H = 0.5$: uncorrelated random walk (pure Brownian motion).
- $H > 0.5$: **persistent** (trending) behavior; positive long-range correlations.
- $H < 0.5$: **anti-persistent** (mean-reverting) behavior; negative long-range correlations.

For self-similar time series, $H$ is directly related to the fractal dimension: $D = 2 - H$.

[[simulation hurst-exponent]]

## The fear-factor model

If stock prices follow a biased random walk, let $q$ be the probability of the stock moving up under normal conditions. Introduce a **collective fear event** with probability $p$ that forces all stocks down simultaneously.

The effective probabilities become:

- Stock goes up: $(1-p)q$.
- Stock goes down: $p + (1-p)(1-q)$.

For a neutral random walk, these must be equal:

$$
(1-p)q = p + (1-p)(1-q) \implies q = \frac{1}{2(1-p)}.
$$

As $p$ increases, the required upward probability $q$ must increase to compensate, creating an asymmetry that mimics the observed negative skewness in stock returns. The variance of returns includes a term proportional to $p$ that represents systematic risk.

## Bet hedging

**Bet hedging** is a strategy where an organism (or investor) sacrifices expected performance to reduce variance, analogous to portfolio diversification.

In a stochastic growth model, consider a population with growth rate $r_t$ drawn from a distribution at each time step. The long-run growth rate is not the arithmetic mean but the **geometric mean**:

$$
\bar{r}_{\text{long-run}} = \langle \ln r_t \rangle = \langle r_t \rangle - \frac{1}{2} \text{Var}(r_t) + \cdots
$$

This **arithmetic-geometric inequality** means that variance always reduces long-run growth. An organism that reduces its variance (by hedging its bets across strategies) can outperform a specialist that maximizes expected growth, especially when the noise is large.

Key insight: the optimal strategy depends on the **noise size** relative to the expected return.

[[simulation bet-hedging]]
