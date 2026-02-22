# Gene Regulatory Networks

> *The universe is not just stranger than we suppose; when it comes to cells, it is stranger than we can suppose -- until we write the equations.*

## Where we are headed

You've built a full toolkit: differential equations, the Hill function, feedback loops, and signaling. Now step back and look at the big picture. Inside every cell, hundreds of genes regulate each other in a vast **network**. How do you make sense of something so complex? The answer: these networks aren't random tangles. They're built from a small set of recurring patterns called **motifs** -- the Lego bricks nature uses over and over.

## Types of regulation

Every regulatory interaction is one of two things:

* **Positive**: $A \rightarrow B$. More A means more B.
* **Negative**: $A \dashv B$. More A means less B.

That's the entire alphabet. The magic is in the combinations.

## The multiplication rule

Given a closed loop, assign $+1$ to each activation arrow and $-1$ to each repression arrow, then multiply around the loop:

$$
A \rightarrow B \rightarrow C \dashv A
$$

gives $(+1)(+1)(-1) = -1$: **negative feedback**. An odd number of repressions makes the loop negative (homeostatic or oscillatory). Even makes it positive (bistable).

## The motif zoo

Certain subpatterns appear far more often in real networks than chance would predict. Evolution discovered these motifs are *useful* and keeps reusing them. Each has its own personality:

### Negative autoregulation -- "the thermostat"

A gene represses itself. The most common motif in *E. coli* -- about half of all TFs do this.

**Personality**: stability and speed. Locks protein to a set point and gets there fast.

### Positive autoregulation -- "the commitment device"

A gene activates itself. With cooperativity, this creates bistability.

**Personality**: memory and irreversibility. *B. subtilis* uses this for competence -- once a cell commits, it stays committed.

### The feed-forward loop -- "the delay timer"

> **Key Equation -- Feed-Forward Loop Logic**
> $$
> X \rightarrow Y \rightarrow Z, \qquad X \rightarrow Z
> $$
> Gene X activates Z through a fast direct path and a slow indirect path via Y; with AND logic, only sustained signals pass through.

The **coherent type-1**: $X$ activates $Z$ two ways -- directly (fast) and through $Y$ (slow). If $Z$ requires *both* $X$ and $Y$ (AND logic), a brief pulse of $X$ does nothing. Only sustained signals get through. It's a spam filter for your genes.

The **incoherent type-1**: $X$ activates $Z$ directly but also activates $Y$ which *represses* $Z$. Result: a pulse -- $Z$ spikes fast then comes back down.

### The toggle switch -- "the memory bank"

$$
U \dashv V, \qquad V \dashv U.
$$

Mutual repression = binary memory. The cell can sit in "high-$U$" or "high-$V$" and remember which. Lambda phage lysis-lysogeny (CI vs. Cro) is the textbook case.

### The single input module -- "the temporal program"

One TF controls a set of targets $Z_1, Z_2, \ldots, Z_n$, each with a different threshold. As the TF rises, targets turn on one by one in order. That's how *E. coli* builds its flagellum.

[[simulation motif-gallery]]

Click through the four motifs to see their step responses. Negative autoregulation reaches steady state fast. Positive feedback shows two trajectories landing at different steady states. The feed-forward loop delays output until the intermediate accumulates. The toggle switch flips and stays -- biological memory.

## Simplifying the equations

When mRNA dynamics are much faster than protein, we can write a single equation for protein. For positive feedback:

$$
\frac{\mathrm{d}P}{\mathrm{d}t} = \frac{\beta \, (P/K)^n}{1 + (P/K)^n} - \gamma \, P,
$$

where $\beta$ absorbs transcription and translation rates. For negative regulation:

$$
\frac{\mathrm{d}P}{\mathrm{d}t} = \frac{\beta}{1 + (P/K)^n} - \gamma \, P.
$$

> *Not cheating -- just separation of timescales. mRNA adjusts so quickly it effectively slaves to the protein level.*

## Graphical analysis

Plot production and degradation on the same graph:

* **Production curve**: S-shaped for activation, decreasing for repression.
* **Degradation line**: $\gamma P$ -- straight through the origin.
* **Where they cross = steady state.** One crossing = monostable. Three = bistable.

[[simulation hill-function]]

Switch between positive and negative regulation. Negative always gives one crossing. Positive with high $n$ -- adjust $\gamma$ until three crossings appear. The outer two are stable; the middle one unstable. That's bistability.

## Why does nature do it this way?

Every principle from earlier lessons -- degradation for speed, the Hill function for switching, feedback for memory and timing -- reappears here as a motif. Nature builds complex behavior from a small reusable toolkit. Learn the motifs and you can read the wiring diagram of any cell.

## Check your understanding

* Given $A \dashv B \dashv C \rightarrow A$: positive or negative feedback? Oscillate or switch?
* Why does the coherent feed-forward loop filter brief signals? What if you replace AND with OR?
* What's the advantage of coordinating flagellar genes through one master regulator?

## Challenge

For the toggle switch with $\dot{U} = \alpha_1/(1+V^n) - U$ and $\dot{V} = \alpha_2/(1+U^n) - V$ ($\alpha_1 = \alpha_2 = 5$, $n = 2$), find the nullclines by setting each derivative to zero. Plot them in the $(U,V)$ plane. Where do they cross? Why does mutual repression create two stable states and one unstable saddle?

## Big ideas

* Networks are built from recurring motifs, each with a personality: thermostat, commitment device, delay timer, memory bank, temporal program.
* The multiplication rule (multiply $\pm 1$ around a loop) instantly tells you positive vs. negative feedback.
* Graphical analysis reveals steady states for any motif without solving equations.

## What comes next

We've assembled the full molecular toolkit. In the final lesson, we put it all together and ask the biggest question: how does a cell *grow*? The answer involves a beautiful resource-allocation problem and a tiny molecule called ppGpp.
