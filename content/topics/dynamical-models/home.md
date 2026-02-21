# Dynamical Models in Molecular Biology

## Why are we here?

Cells are tiny factories that make decisions with molecules that bump around randomly. A single *E. coli* bacterium — two micrometers long, swimming through your gut — can sense a gradient of food so faint that it amounts to one extra molecule per cell volume per micron. It can remember what happened a minute ago. It can flip a genetic switch and commit to a new lifestyle. And it can double itself in twenty minutes by allocating its limited protein budget with a precision that would make an economist weep.

How does it do all of this? Not with a brain, not with a blueprint, but with differential equations, noise, feedback, and a handful of molecular tricks that evolution has been polishing for four billion years.

This course gives you the tools to understand — and simulate — every one of those tricks. We start with the simplest equation you can write down (molecules appearing and disappearing), and we build, lesson by lesson, until you can watch a bacterium decide how many ribosomes to make, and understand *why* it is exactly the right number.

Here is the arc: smooth deterministic equations show you the clockwork. Then we shake the clockwork and discover that noise is not a bug but a feature. We add feedback and suddenly cells can remember, decide, and keep time. We zoom out to networks and signals. And at the end, we put it all together in a growing bacterium that turns physics into life.

## What you will learn

- **Physics provides the framework**: differential equations, stochastic processes, and steady-state analysis.
- **Biology provides the systems**: genes, proteins, regulatory circuits, and growing cells.
- **Mathematics provides the rigor**: exact solutions, stability analysis, and parameter estimation.

We focus on simple systems — often bacterial — but the principles apply broadly across all of biology.

## Key mathematical ideas

- Ordinary differential equations for production and degradation kinetics.
- The Hill function as a universal model for cooperative binding.
- Stochastic simulation (Gillespie algorithm) for single-cell noise.
- Fixed-point analysis and bifurcation diagrams for feedback circuits.
- Resource allocation models for bacterial growth.

## Cast of characters

You will meet these molecules and systems throughout the course. Think of them as the recurring cast of a story — each one has a personality and a job:

| Character | Job description |
|---|---|
| **RNA polymerase** | The scribe. Reads DNA and writes mRNA copies. |
| **Ribosome** | The factory floor. Reads mRNA and builds protein, one amino acid at a time. |
| **mRNA** | The shopping list. Carries instructions from the DNA cookbook to the ribosome kitchen. Short-lived and disposable. |
| **LacI** | The classic bouncer. Represses the *lac* operon in *E. coli* — sits on the promoter and blocks the scribe. |
| **GFP / CFP / YFP** | The spies. Fluorescent reporter proteins that let us *see* gene expression in single cells. |
| **CheY, CheA, CheR, CheB** | The navigation crew. The signaling proteins of bacterial chemotaxis — they tell the flagellar motor to run or tumble. |
| **CI and Cro** | The dueling regulators. They fight over phage lambda's life-or-death decision: hide in the genome, or kill the host. |
| **ppGpp** | The alarm bell. Tells the cell to stop making ribosomes when nutrients run low. The master switch of bacterial resource allocation. |
| **ComK** | The gambler. Master regulator of competence in *B. subtilis* — noisy positive feedback flips a rare fraction of cells into a DNA-absorbing state. |
| **sRNA** | The molecular Post-it note. Small RNA that sticks to mRNA and says "delete me." |

## Visual and Simulation Gallery

[[simulation binomial-poisson-comparison]]

## Cheat sheet of analogies

When you get lost in the symbols, come back here:

| Concept | Analogy |
|---|---|
| Bathtub equation ($\dot{n} = k - \Gamma n$) | Water flowing into a bathtub with a leaky drain. Steady state = when inflow matches outflow. |
| mRNA | A shopping list sent from the DNA cookbook to the ribosome kitchen. Short-lived — gets recycled fast. |
| Protein | The actual dish that the kitchen (ribosome) produces from the shopping list (mRNA). Lasts much longer. |
| Two-timescale filtering | A radio speaker smoothing out static from the antenna. mRNA jitters fast; protein follows slowly. |
| Hill function | A light switch. One finger barely moves it, but $n$ fingers snap it on. The Hill coefficient $n$ is the number of fingers. |
| Dissociation constant $K$ | The halfway point of the switch — the concentration where the gene is half on. |
| Negative feedback | A thermostat. If the room gets too hot, it turns off the heater. If too cold, it turns it back on. |
| Positive feedback | A microphone pointed at a speaker. Once it starts, the signal amplifies itself until it saturates. |
| Bistability | A light switch with two stable positions — up and down — and an unstable balance in the middle. |
| Repressilator | Three friends chasing each other in a circle: A represses B, B represses C, C represses A. Nobody can rest. |
| Chemotaxis adaptation | Your eyes adjusting to a dark room. After a while the brightness dial resets so you can still detect tiny changes. |
| CheY-P | The tumble messenger. When it is abundant, the bacterium tumbles; when it is scarce, the bacterium runs. |
| ppGpp | The factory alarm bell. When raw materials run short, it rings and the ribosome factories slow down. |
| Feed-forward loop | A delay timer. The signal must persist long enough to pass through two paths before the gene turns on. |
| Ribosome allocation ($\phi_R$) | Deciding what fraction of your budget to spend on building more factories versus running the ones you have. |

> **Figure: Course concept map.** A one-page diagram showing the flow of ideas through the course: bathtub equation $\to$ two timescales $\to$ probability & mutations $\to$ noise $\to$ Hill function $\to$ feedback loops $\to$ signals $\to$ network motifs $\to$ growth physiology. Arrows show dependencies between concepts.

## What did we just build?

Look at what you now own: a complete mental model of a living cell deciding, remembering, growing, and evolving. You started with a bathtub and ended with a bacterium that allocates its protein budget to maximize growth — and you understood every step of the way *why* it works.

You can now read a quantitative biology paper and follow the equations. You can simulate gene circuits on your laptop. You can look at a noisy single-cell dataset and tell whether the noise is intrinsic or extrinsic, bursty or Poisson. You can draw a feedback loop and predict whether it will oscillate or switch.

That is not small. Go use it on the next paper you read, the next dataset you analyze, the next conversation you have about how life actually works.
