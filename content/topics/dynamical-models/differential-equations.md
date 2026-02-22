# Differential Equations in a Nutshell

> *The universe is not just stranger than we suppose; when it comes to cells, it is stranger than we can suppose -- until we write the equations.*

## Where we are headed

We need a language for talking about how things change over time. That language is differential equations. One simple idea -- *the rate of change equals what flows in minus what flows out* -- is enough to explain how molecules accumulate, decay, and reach a steady state inside a living cell. Everything else in this course builds on top of this.

## The bathtub picture

You're filling a bathtub. Water flows in at constant rate $k$ (molecules per minute) and drains out proportional to how full it is: $\Gamma n$. The net rate of change is production minus loss:

> **Key Equation -- The Bathtub Equation**
> $$
> \dot{n} = k - \Gamma n
> $$
> The rate of change of molecule number equals production minus degradation: what flows in minus what flows out.

At steady state the tub is neither rising nor falling, so $\dot{n} = 0$. Solve: $n_\mathrm{ss} = k / \Gamma$. Every steady state in this entire course looks like this.

And here's the gorgeous part -- this is *exactly* how molecules work inside a cell. mRNA is produced by the transcription machinery and degraded proportional to how much is present. The steady-state concentration is simply the ratio of production to degradation. Nature finds this balance automatically. Let's derive it step by step.

## Creation: molecules appearing at a constant rate

Suppose mRNA is created at a constant rate $k$ (molecules per unit time). After a short time $\Delta t$, the count goes from $n(t)$ to

$$
n(t+\Delta t) = n(t) + k \, \Delta t.
$$

> *New count = old count + however many were made during the interval.*

Rearranging and taking the limit $\Delta t \rightarrow 0$:

$$
\frac{\mathrm{d} n(t)}{\mathrm{d} t} = k.
$$

The solution is $n(t) = n(0) + kt$ -- a straight line. Molecules pile up forever. Clearly something's missing.

## Degradation: molecules falling apart

Now suppose molecules are degraded at rate $\Gamma$ per unit time. The number destroyed in $\Delta t$ is $\Gamma \, n(t) \, \Delta t$ -- it depends on how many are around:

$$
\frac{\mathrm{d} n(t)}{\mathrm{d} t} = -\Gamma \, n(t).
$$

> *The more molecules you have, the faster they vanish.*

The solution is $n(t) = n(0) \, e^{-\Gamma t}$ -- exponential decay. You can also think of this as the **half-life** $t_{1/2} = \ln 2 / \Gamma$, the time for half the molecules to be gone, or the **time constant** $\tau = 1/\Gamma$, the time to drop to about 37%.

## The full picture: creation and degradation together

Combine both. This is the bathtub equation:

$$
\frac{\mathrm{d} n(t)}{\mathrm{d} t} = k - \Gamma \, n(t).
$$

> *Rate of change = what's being made minus what's falling apart.*

At **steady state** nothing changes, so set the left side to zero:

$$
0 = k - \Gamma \, n_\mathrm{ss} \quad \Longrightarrow \quad n_\mathrm{ss} = \frac{k}{\Gamma}.
$$

> *Steady-state molecules = production rate divided by degradation rate. That's it. This beautifully simple result shows up everywhere in biology.*

Start with zero molecules? The system climbs toward $n_\mathrm{ss}$ on a timescale set by $1/\Gamma$. Start above? The excess decays on the same timescale.

[[simulation bathtub-dynamics]]

Slide the production and degradation knobs yourself. Watch how fast degradation makes the system snap to its new steady state -- that's why cells keep mRNA on a short leash.

## Why does nature do it this way?

You might wonder why a cell would waste energy destroying its own mRNA. The answer is speed. A cell that only produces molecules can never reduce their levels -- it's stuck. But a cell that both produces *and* degrades can change its steady state simply by adjusting the production rate. The faster the degradation, the quicker the response. That's why many bacterial mRNAs have half-lives of just a few minutes.

## Check your understanding

* Double $k$ while keeping $\Gamma$ fixed -- what happens to steady state?
* An mRNA has a half-life of 3 minutes. What is $\Gamma$?
* Why does faster degradation mean faster response, even though it seems wasteful?

## Challenge

A gene starts producing mRNA at $k = 10$ molecules/min from zero, with $\Gamma = 0.2$/min. Without solving the ODE, guess the steady state and the approximate time to get there. Then solve $\dot{n} = k - \Gamma n$ and check. How close were you?

## Big ideas

* The bathtub equation $\dot{n} = k - \Gamma n$ is the foundation of almost every model in this course.
* Steady state is where production balances degradation: $n_\mathrm{ss} = k / \Gamma$.
* The degradation rate $\Gamma$ sets the response time -- fast turnover means the cell can change its mind quickly.

## What comes next

Real cells make proteins from mRNA, introducing a second variable and two very different timescales -- that's where we go next.
