# Econophysics

Remember the sandpile from the Self-Organized Criticality section? You drop grains one at a time, and most of the time nothing happens, but occasionally a massive avalanche rearranges the whole pile. The same mathematics shows up in stock prices.

Remember the scale-free networks from the Networks section? A few hubs dominate, and the degree distribution follows a power law. The same pattern appears in the distribution of stock returns: small fluctuations are common, but extreme moves (crashes and rallies) are far more frequent than a normal distribution would predict.

Remember the Ising model from the very beginning? Spins interact with their neighbors and collectively decide to align — a phase transition. Traders interact with each other (through news, rumors, herding) and collectively decide to sell — a market crash.

**Econophysics** applies the methods of statistical mechanics and complex systems to financial markets. The central discovery is that markets are not the well-behaved random walks that classical finance assumes. They are complex systems with heavy tails, long-range correlations, and avalanche-like dynamics.

## Brownian motion and stock prices

The simplest model treats log-returns as a random walk. Let $S_t$ be the closing price and $x_t = \log(S_t)$. The variance at lag $\tau$ is:

$$
\text{Var}(\tau) = \langle (x_{t+\tau} - x_t)^2 \rangle.
$$

For **geometric Brownian motion** (the foundation of the Black-Scholes model), variance grows linearly with lag:

$$
\text{Var}(\tau) \propto \tau.
$$

This would mean stock returns are uncorrelated, Gaussian, and well-behaved. It is a beautiful theory. It is also wrong.

[[simulation stock-variance]]

Run this simulation with real stock data and compare it with the prediction from geometric Brownian motion. You will see two major deviations: (1) **heavy tails** — large moves are far more frequent than a Gaussian predicts (the famous "fat tails"), and (2) **volatility clustering** — large changes tend to follow large changes, and calm periods follow calm periods. The market has memory.

## The Hurst exponent

The **Hurst exponent** $H$ measures long-range dependence in time series. For the rescaled range $R/S$ of a time series with $n$ data points:

$$
\frac{R}{S} \sim C \, n^H \quad \text{as } n \to \infty,
$$

where $C$ is a constant.

The value of $H$ reveals the nature of correlations:

- $H = 0.5$: uncorrelated random walk (pure Brownian motion). Each step is independent of the past.
- $H > 0.5$: **persistent** (trending) behavior. An up move is more likely to be followed by another up move. Positive long-range correlations.
- $H < 0.5$: **anti-persistent** (mean-reverting) behavior. An up move is more likely to be followed by a down move. Negative long-range correlations.

For self-similar time series, $H$ is directly related to the fractal dimension (from the Percolation and Fractals section): $D = 2 - H$. A Brownian motion trace has fractal dimension $1.5$; a persistent time series is smoother ($D < 1.5$); an anti-persistent one is rougher ($D > 1.5$).

[[simulation hurst-exponent]]

Estimate the Hurst exponent from data in this simulation. Real financial time series typically show $H$ close to $0.5$ for the returns themselves (they are nearly uncorrelated), but $H$ significantly above $0.5$ for the absolute returns or volatility (volatility is persistent — this is the clustering effect).

## The fear-factor model

Here is a simple model that captures the asymmetry in stock returns. Let $q$ be the probability of the stock moving up under normal conditions. Now introduce a **collective fear event** with probability $p$ that forces all stocks down simultaneously — think of a market panic, a financial crisis, a sudden loss of confidence.

The effective probabilities become:

- Stock goes up: $(1-p)q$.
- Stock goes down: $p + (1-p)(1-q)$.

For a neutral random walk (equal probability of up and down):

$$
(1-p)q = p + (1-p)(1-q) \implies q = \frac{1}{2(1-p)}.
$$

As the fear probability $p$ increases, the required upward probability $q$ must increase to compensate. This creates an asymmetry that mimics the observed **negative skewness** in stock returns: the market rises slowly most of the time (to compensate for the fear premium) but crashes quickly and dramatically when the collective fear event hits.

The variance of returns includes a term proportional to $p$ that represents **systematic risk** — risk that cannot be diversified away because it affects everyone simultaneously.

## Bet hedging

**Bet hedging** is a strategy where an organism (or investor) sacrifices expected performance to reduce variance. It is the mathematical foundation of portfolio diversification.

In a stochastic growth model, consider a population with growth rate $r_t$ drawn from a distribution at each time step. The long-run growth rate is not the arithmetic mean but the **geometric mean**:

$$
\bar{r}_{\text{long-run}} = \langle \ln r_t \rangle = \langle r_t \rangle - \frac{1}{2} \text{Var}(r_t) + \cdots
$$

This **arithmetic-geometric inequality** means that variance *always* reduces long-run growth. An organism (or investor) that reduces its variance — by hedging its bets across strategies — can outperform a specialist that maximizes expected growth, especially when the noise is large.

[[simulation bet-hedging]]

Try the simulation: compare a "specialist" strategy (high expected return, high variance) with a "diversified" strategy (lower expected return, lower variance). Run both for many time steps. The specialist might win in the short run, but the diversifier almost always wins in the long run. Variance is not just risk — it is a *drag* on growth.

The key insight: the optimal strategy depends on the **noise size** relative to the expected return. In a calm, predictable world, specialize. In a noisy, uncertain world, diversify. The mathematics is the same whether you are a bacterium hedging against environmental fluctuations or a portfolio manager hedging against market crashes.

> **Key Intuition.** Financial markets are complex systems, not well-behaved random walks. The same power laws that appear in sandpiles, percolation clusters, and scale-free networks show up in the distribution of stock returns. Heavy tails, volatility clustering, and collective fear events are all signatures of a system near criticality. Nature reuses her tricks — and the mathematics of complex physics applies far beyond physics.

> **Challenge.** Here is a bet: I flip a fair coin. Heads, your wealth increases by 50%. Tails, it decreases by 40%. The expected return per flip is $0.5 \times 1.5 + 0.5 \times 0.6 = 1.05$ — a 5% expected gain! Sounds great. But compute the geometric mean: $(1.5 \times 0.6)^{1/2} = 0.949$. You *lose* about 5% per flip in the long run. This is the arithmetic-geometric inequality in action. Would you take this bet? Why or why not?
