# Differential Equations in a Nutshell

## Where we are headed

This is our first lesson, and we need a language for talking about how things change over time. That language is differential equations. By the end of this lesson you will see that one simple idea — *the rate of change equals what flows in minus what flows out* — is enough to explain how molecules accumulate, decay, and reach a steady state inside a living cell. Everything else in this course builds on top of this.

## The bathtub picture

Imagine you are filling a bathtub. Water flows in at constant rate $k$ (molecules per minute) and drains out at rate proportional to how full it is: $\Gamma n$. The net rate of change is production minus loss:

$$
\dot{n} = k - \Gamma n.
$$

At steady state the tub is neither rising nor falling, so $\dot{n} = 0$. Solve: $n_\mathrm{ss} = k / \Gamma$. Every steady state in this entire course is of this form.

This is exactly how molecules work inside a cell. mRNA is produced at some rate by the transcription machinery, and it is degraded at a rate proportional to how much is present. The steady-state concentration is simply the ratio of production to degradation. Nature finds this balance automatically. Let us now derive this step by step.

> **Figure: Bathtub cartoon.** A bathtub with a faucet labelled "production $k$" and a drain labelled "degradation $\Gamma$." The water level represents the molecule count $n$. An arrow pointing to the water surface is labelled "$1/\Gamma$ = response time."

> **Figure: Approach to steady state.** Time-course plot with two curves: one starting at $n = 0$ rising toward the horizontal steady-state line $n_\mathrm{ss}$, and one starting above $n_\mathrm{ss}$ decaying down to it. Vertical arrows on both curves labelled "time constant $1/\Gamma$."

## Creation: molecules appearing at a constant rate

Suppose molecules (say, mRNA) are created at a constant rate $k$ — the "birth rate" — (molecules per unit time). After a short time $\Delta t$, the number of molecules goes from $n(t)$ to

$$
n(t+\Delta t) = n(t) + k \, \Delta t.
$$

> *In words: the new count equals the old count plus however many were made during that interval.*

Rearranging and taking the limit $\Delta t \rightarrow 0$ gives us our first differential equation:

$$
\frac{\mathrm{d} n(t)}{\mathrm{d} t} = k.
$$

> *This says: the number of molecules increases at a constant rate, no matter how many are already there.*

The solution is simply $n(t) = n(0) + kt$ — a straight line. Molecules pile up forever. Clearly something is missing.

## Degradation: molecules falling apart

Now suppose molecules are degraded at a rate $\Gamma$ — the "death rate constable" — (per unit time). The number destroyed in a short interval $\Delta t$ is $\Gamma \, n(t) \, \Delta t$ — it depends on how many molecules are present. Following the same logic:

$$
\frac{\mathrm{d} n(t)}{\mathrm{d} t} = -\Gamma \, n(t).
$$

> *This says: the more molecules you have, the faster they disappear.*

The solution is $n(t) = n(0) \, e^{-\Gamma t}$ — exponential decay. You can also think of this in terms of the **half-life** $t_{1/2} = \ln 2 / \Gamma$, the time it takes for half the molecules to be gone. Or equivalently, the **time constant** $\tau = 1/\Gamma$, which is the time for the number to drop to about 37% of its starting value.

## The full picture: creation and degradation together

Now combine both processes. This is the bathtub equation:

$$
\frac{\mathrm{d} n(t)}{\mathrm{d} t} = k - \Gamma \, n(t).
$$

> *In words: the rate of change equals what's being made minus what's falling apart.*

At **steady state** nothing is changing, so we set the left side to zero:

$$
0 = k - \Gamma \, n_\mathrm{ss} \quad \Longrightarrow \quad n_\mathrm{ss} = \frac{k}{\Gamma}.
$$

> *The steady-state number of molecules is just the production rate divided by the degradation rate. That's it. This beautifully simple result shows up everywhere in biology.*

If you start with zero molecules, the system climbs toward $n_\mathrm{ss}$ on a timescale set by $1/\Gamma$ — fast degradation means the system reaches steady state quickly. If you start above $n_\mathrm{ss}$, the excess decays away on the same timescale.

## Why does nature do it this way?

You might ask: why does a cell bother degrading its own mRNA? It costs energy to make it, so why destroy it? The answer is speed. A cell that only produces molecules (no degradation) can never reduce their levels — it is stuck. But a cell that both produces and degrades can change its steady state simply by adjusting the production rate. The faster the degradation, the quicker the response. This is why many mRNAs in bacteria have half-lives of just a few minutes.

## Check your understanding

- If you double the production rate $k$ while keeping $\Gamma$ fixed, what happens to the steady-state level?
- A certain mRNA has a half-life of 3 minutes. What is its degradation rate $\Gamma$?
- Why does faster degradation lead to faster response, even though it seems wasteful?

## Challenge

Suppose a gene starts producing mRNA at rate $k = 10$ molecules per minute, starting from zero, with degradation rate $\Gamma = 0.2$ per minute. Without solving the differential equation, try to guess the steady-state level and the approximate time to get there. Then solve $\dot{n} = k - \Gamma n$ and check your guess. How close were you?

## Big ideas

- **The bathtub equation** $\dot{n} = k - \Gamma n$ is the foundation of almost every model in this course.
- **Steady state** is where production balances degradation: $n_\mathrm{ss} = k / \Gamma$.
- **The degradation rate $\Gamma$ sets the response time** — fast turnover means the cell can change its mind quickly.

## What comes next

We now have the basic language of change: production, degradation, and the steady state that emerges from their balance. But real cells are noisy — molecules are made and destroyed one at a time, not in smooth continuous flows. In the next lesson, we shake our beautiful deterministic equations and discover that the randomness is not a nuisance but a feature. Get ready: the steady state you just learned to love is about to become a lie.
