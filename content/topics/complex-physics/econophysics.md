# Econophysics

Remember the sandpile? You drop grains one at a time, mostly nothing happens, then occasionally a massive avalanche rearranges the whole pile. The same mathematics shows up in stock prices.

Remember scale-free networks? A few hubs dominate, degree distribution follows a power law. Same pattern in stock returns: small fluctuations are common, but extreme moves -- crashes and rallies -- are far more frequent than a normal distribution would predict.

Remember the Ising model? Spins interact with neighbors and collectively decide to align -- a phase transition. Traders interact through news, rumors, herding, and collectively decide to sell -- a market crash.

**Econophysics** applies statistical mechanics to financial markets. The central discovery: markets aren't the well-behaved random walks that classical finance assumes. They're complex systems with heavy tails, long-range correlations, and avalanche-like dynamics.

## Big Ideas

* Real prices look like a random walk on the surface, but the volatility -- the size of the steps -- clusters like earthquakes. The market has memory in its nervousness, not in its direction.
* The arithmetic-geometric inequality is the deepest insight here: variance isn't just "risk" in an abstract sense -- it's a literal drag on long-run growth, which is why diversification is mathematically optimal, not just psychologically comforting.
* The fear-factor model captures market asymmetry in one parameter: crashes are fast and correlated, rallies are slow and diffuse, because collective panic synchronizes more easily than collective optimism.

## Brownian Motion and Stock Prices

The simplest model treats log-returns as a random walk. Let $S_t$ be the price and $x_t = \log(S_t)$. The variance at lag $\tau$:

$$
\text{Var}(\tau) = \langle (x_{t+\tau} - x_t)^2 \rangle.
$$

For **geometric Brownian motion** (foundation of Black-Scholes), $\text{Var}(\tau) \propto \tau$. Returns would be uncorrelated, Gaussian, well-behaved. Beautiful theory. Also wrong.

[[simulation stock-variance]]

Run this with real data and compare with geometric Brownian motion. You'll see two major deviations: **heavy tails** (large moves far more frequent than Gaussian) and **volatility clustering** (large changes follow large changes, calm follows calm). The market has memory.

## The Hurst Exponent

The **Hurst exponent** $H$ measures long-range dependence. For the rescaled range $R/S$ of a time series:

$$
\frac{R}{S} \sim C \, n^H.
$$

* $H = 0.5$: uncorrelated random walk.
* $H > 0.5$: **persistent** (trending). Up follows up.
* $H < 0.5$: **anti-persistent** (mean-reverting). Up follows down.

For self-similar series, $H$ connects to fractal dimension: $D = 2 - H$.

[[simulation hurst-exponent]]

Log-returns themselves are nearly uncorrelated ($H \approx 0.5$), but absolute returns show persistent long-range correlations ($H \approx 0.7$-$0.8$). The returns are unpredictable, but their *magnitude* has long memory. That's volatility clustering in one number.

## The Fear-Factor Model

Let $q$ be the probability of moving up under normal conditions. Introduce a **collective fear event** with probability $p$ that forces all stocks down simultaneously -- market panic.

For a neutral random walk:

$$
(1-p)q = p + (1-p)(1-q) \implies q = \frac{1}{2(1-p)}.
$$

As fear $p$ increases, the required $q$ increases to compensate. This creates **negative skewness**: the market rises slowly most of the time but crashes quickly and dramatically when panic hits. The variance includes systematic risk -- risk that can't be diversified away because it hits everyone at once.

## Bet Hedging

**Bet hedging** is the mathematical foundation of diversification. In a stochastic growth model with growth rate $r_t$, the long-run growth rate isn't the arithmetic mean but the **geometric mean**:

$$
\bar{r}_{\text{long-run}} = \langle \ln r_t \rangle = \langle r_t \rangle - \frac{1}{2} \text{Var}(r_t) + \cdots
$$

This **arithmetic-geometric inequality** means variance *always* reduces long-run growth. An organism (or investor) that reduces variance by hedging across strategies can outperform a specialist that maximizes expected growth, especially when noise is large.

[[simulation bet-hedging]]

Compare a "specialist" (high expected return, high variance) with a "diversifier" (lower expected return, lower variance). Run both for many steps. The specialist might win short-term, but the diversifier almost always wins long-term. Variance isn't just risk -- it's a *drag* on growth.

The math is the same whether you're a bacterium hedging against environmental shifts or a portfolio manager hedging against crashes. In a calm, predictable world, specialize. In a noisy, uncertain world, diversify.

## What Comes Next

With econophysics, the arc of complex physics comes full circle. We started with [Statistical Mechanics](statisticalMechanics) -- counting states. We built up: [Metropolis Algorithm](metropolisAlgorithm) for simulation, [Phase Transitions](phaseTransitions) for collective ordering, [Mean-Field Results](meanFieldResults) for theory, the [Transfer Matrix](transferMatrix) for exact solutions, [Critical Phenomena](criticalPhenomena) for universality, [Percolation](percolation) and [Fractals](fractals) for geometric criticality, [Self-Organized Criticality](selfOrganizedCriticality) for systems that tune themselves to the edge, [Networks](Networks) for interaction topology, and [Agent-Based Models](agentbased) for emergence.

The unifying theme: complex collective behavior -- phase transitions, power laws, fractals, emergent order -- arises not from special ingredients but from simple interactions among many components. Whether those components are spins, sand grains, network nodes, birds, or traders, the mathematics is the same. That's the deepest lesson of complex physics.

## Check Your Understanding

1. Real stock returns have $H \approx 0.5$ (uncorrelated), but absolute returns have $H > 0.5$ (persistent). What does it mean that prices are unpredictable but volatility has memory?
2. In bet hedging, variance reduces long-run growth by $\frac{1}{2}\text{Var}(r_t)$. If two strategies have the same arithmetic mean return but different variances, which wins long-term, and why?

## Challenge

Download a real stock-index time series (e.g. S&P 500 daily closes). Compute daily log-returns $x_t = \log(S_{t+1}/S_t)$. Test the Gaussian assumption: plot the empirical distribution against a fitted Gaussian and compute the kurtosis (Gaussian has kurtosis 3). Compute the autocorrelation of $|x_t|$ at lags 1, 5, 10, 20, 50 days. Estimate the Hurst exponent of $|x_t|$ using rescaled-range analysis for window sizes $n = 10, 20, 50, 100, 200$. Compare your findings to the random-walk prediction ($H = 0.5$, kurtosis $= 3$, zero autocorrelation). What do the deviations tell you about what a realistic model needs?
